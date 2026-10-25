# Private source-policy integration

## Current checkpoint

Snapshot: 2026-10-25. This unfinished LOCAL WIP is based on stored shared
opset-21 checkpoint **`b4f8b16e`**; **`fe214590`** remains the last fully green
checkpoint. The later deployment-test source snapshot is **`c9f5812b`** and the
applied inventory-test source snapshot is **`b7eceac8`**. Results below retain
their own source epochs.
Public routes and defaults remain **LEGACY**; the full **B–F** goal remains open.

| Current gate | Actual source-bound result | Scope and remaining work |
| --- | --- | --- |
| Shared opset-21 support, source `12ee5c05`, local Python 3.12 and 3.11 | **520 PASS each**, identical unique selections across six complete owners, zero failures/errors/skips; PRIMARY **90.448 s**, SECONDARY **93.671 s**, source/Git held | Includes **65 new cases** using real private SOURCE export, saved opset-21 artifacts/full checker, native eager/fresh-JIT references and both CPU optimization modes. This qualifies the finite samples and unchanged guards; deliberately failed subnormal target diagnostics remain failed-target evidence. |
| Four unchanged original AttentionCore cells, source `12ee5c05` | Both raw-x64 caller settings on both local stacks: source reference, artifact, schema and lowering audit **VALIDATED**; original raw callback **FAILED**; structure/runtime/numerics **UNVERIFIED** | These cells progressed past admission. Original callback, `None` criteria and row contract `ea6d8f6c7fab785a98941d54722b2926789b2edd59d74e81b40992e64d304e9d` remain unchanged. CLI completion and the small attention fixtures do not establish original-model acceptance. |
| Test-only deployment portability, source `c9f5812b`, local Python 3.12 and 3.11 | **142 PASS each**, identical unique four-owner selections, zero failures/errors/skips; PRIMARY **92.479 s**, SECONDARY **78.234 s**, source/Git held | Actual matching-host strict numerical branches, real cross-ABI observations and CPU-only negative controls ran under both optimization modes. Original subnormal numerical **FAILED** remains failed-target evidence. No production target, observer, criterion or public-default change. |
| Optional-discovery fixture, source snapshot `b7eceac8`, normal local Python 3.12 and 3.11 | **49 PASS each**, identical unique complete-owner selections, zero failures/errors/skips; PRIMARY **10.295 s**, SECONDARY **6.528 s**, source/Git held | Normal optional sources are unconfigured; two new isolated family cases check configured external discovery and local preservation. Under the exact same deliberately incomplete external source, the original absence test changes from **1 FAIL** to **1 PASS**. This is local red/green evidence, not a recovered CI traceback, optional-model acceptance or a production inventory/baseline change. |
| Retained invalid global optional-source environment, source snapshot `b7eceac8` | Complete SECONDARY owner: **48 PASS / 1 FAIL**, zero errors/skips | A safe raw-identity query proves the sole extra row is actual `MaxDiffusion_Environment_Error` from the deliberately invalid global fixture; the unchanged 1,852-row baseline guard correctly rejects it. This remains failed environment evidence, not an accepted model or a product regression. |
| Remote CI run `40210233250`, stored head `b4f8b16e` | Four support jobs **PASS**; four full Python 3.11/3.12/3.13/3.14 jobs **running** at ROOT's recorded observation | This remote run does not contain the later uncommitted test/CI-logging fixes. It is not current-cut or complete CI acceptance. |
| Historical CI run `39908291998` | Four support jobs **PASS**, including minimum Python 3.11 **465 tests** and normalization Python 3.12 **102 tests**; full Python 3.11/3.12/3.13 jobs **FAILED/incomplete**, exit **143**, with **21/22/54 earlier failure markers**, respectively; full Python 3.14 completed with **145 FAIL / 13,465 PASS / 11 SKIP / 215 ERROR** | Executed merge `859c266` into `fae178`. Original failure markers remain distinct from later grounded local reproductions. Python 3.14 also retains lazy-annotation and native-capture compatibility failures. Neither interrupted failures nor completed compatibility failures are relabeled as infrastructure-only or full green. |
| Current local closeout | **PENDING actual quality/docs/hooks/staged review** | Record terminal current-cut outcomes in the normal commit record. No local complete-suite green result is claimed. |

The current portability changes are confined to test fixtures and test-job
verbosity (`PYTEST_ADDOPTS='-v --tb=short'`). They choose explicit existing reviewed
cross-export declarations without changing production registration, observation
or numerical criteria. Matching actual targets retain all original comparisons;
unreviewed actual targets must produce exact rejection before sample execution,
with runtime-load **FAILED**, no outputs and numerics **UNVERIFIED**. Provider,
thread, configuration, graph-boundary and matrix-binding guards remain strict.
Optional discovery remains separate from source audit and numerical acceptance;
production inventory, baseline rows, optional counts and model selection are
unchanged.

The shared change admits only FP32/FP64 `dot_general`, transpose, `reduce_max`,
exp and `stop_gradient` in the existing opset-21 operation/type intersection,
and extends Max21 to FP32/FP64. Max21 FP32 reuses the existing signed-zero
selection algorithm; Min21 FP64 and all owning native type/result, dimension,
symbolic-scope, precision and placement checks remain unchanged. Opset18,
unsupported contraction dtypes and HIGH/HIGHEST/algorithm gates retain their
existing contracts. This is shared implementation, without a row allowlist or
changes to the original probe, inventory, callbacks or numerical criteria.

Python **3.11 and 3.12** are primary user cohorts and each requires its actual CI
stack. Local PRIMARY Python 3.12/JAX 0.12.0.1 and SECONDARY Python 3.11/JAX 0.10.2
passes do not replace the locked Python 3.12/JAX 0.10.2 CI cohort or the separate
minimum stack. Resolve the original AttentionCore structural contract and then
qualify its runtime/numerical behavior; retain the interrupted **14,053 PASS /
2 FAIL / 18 unexecuted** full attempt and every older failure epoch below.
Current inventory/probe reconciliation, fixed-source and opaque/metadata-free/
manual obligations, scientific FP64/low-precision workflows and coherent public
validation/preparation/export cutover remain open. The four new observations do
not add closures to the frozen caller map.

Current evidence: `root-deployment-portability-paired-native-proof-20261025-v1.json`;
`root-deployment-portability-native-branch-evidence-20261025-v1.json`;
`root-inventory-environment-paired-native-proof-20261025-v1.json`;
`root-ci312-optional-environment-original-red-proof-20261025-v1.json`;
`root-inventory-configured-registration-diff-20261025-v1/identities.json`;
`source-ci312-inventory-environment-fixed-20261025-v1-secondary/receipt.json`;
`source-inventory-environment-root-application-20261025-v1/source-inputs.json`;
`source-opset21-shared-attention-root-application-v2/source-inputs.json`;
`root-opset21-shared-attention-paired-result-v2.json`;
`source-opset21-shared-attention-v2-primary/receipt.json`;
`source-opset21-shared-attention-v2-secondary/receipt.json`;
`root-attentioncore-after-shared-opset21-summary-v2.json`;
`root-pr277-ci311-terminal-v1/receipt.json`.

### Retained 793 whole-entry checkpoint record

The entire previous current record below is preserved verbatim at its original
source cut. The new record above supersedes its then-current and pending wording
without relabeling historical outcomes, criteria or receipts.

Snapshot: 2026-10-23. This unfinished LOCAL WIP follows **`d0835d02`**;
**`fe214590`** remains the last fully green checkpoint. The applied whole-entry
SOURCE extension is source **0e9190da** (**1,158 source pins**). Public routes,
defaults and `enable_double_precision` remain **LEGACY**.

| Current gate | Actual source-bound result | Scope and remaining work |
| --- | --- | --- |
| CI failure at `d52`, Python 3.11 and 3.12 | **FAILED typing**: optional `jax._src.flattree` static import and BF16 `no-any-return` | Historical failures remain. The first cast successor passed SECONDARY typing but failed PRIMARY with `redundant-cast`; it is not relabeled. |
| Separate CI repair preceding `d083` | **572 mypy files PASS on each stack**; **173 tests PASS each**, zero failures/errors/skips | The three complete owners cover BF16 samples, Scan metadata and JAX compatibility. The final ndarray local retains the exact BF16 copy; typed lazy FlatTree loading preserves old-stack early return and native ownership/error checks. Runner-input snapshot `37802360` is distinct from application/commit source snapshots. These typing logs contain no new RNG-trace claim. |
| Stored CI repair `d0835d02` | Actual normal commit completed its guarded command and all seven normal hooks; source files held | CI run **39908291998** at the last ROOT observation had browser, both normalization and minimum-Python-3.11 jobs PASS; four full jobs were still running. No complete green matrix is claimed, and these jobs do not complete the interrupted local suite. |
| Whole-entry four-setting SOURCE facade, source `0e9190da` | **315 PASS each**, identical unique selections; PRIMARY **18.601 s**, SECONDARY **16.065 s**, zero failures/errors/skips, source/Git held | Six complete owners include the original eight tests and 36 new early/final cases, LEGACY controls, global/local/unset checks and proto/IR/file returns. Cleanup completed without guard stops; peak owned RSS **540,049,408 / 697,696,256 bytes**. This is private-boundary qualification, not full-suite or numerical acceptance. |
| This WIP's final closeout | **PENDING actual current quality/docs/hooks/staged review** | Record terminal outcomes in the normal commit record. Prior commit checks do not establish this later cut's quality or full migration acceptance. |

The existing caller-preservation helper now encloses all in-function SOURCE
facade work after the pure policy identity check: logging/repr, input shape/dtype
protocols and iteration, names, mappings and paths, conversion, final naming,
report construction and saving/return. The original late preservation scope is
retained, preserving restoration before finalization. Exactly **x64, matrix
precision, dtype promotion and rank promotion** restore both global and
local/unset values after success or exceptions, without hiding updates during
the call. This is not arbitrary JAX-state restoration or protection of argument
evaluation before Python enters the function. The public LEGACY branch and its
native local-over-global flag precedence are unchanged.

The prior full attempt remains **INTERRUPTED: 14,053 PASS / 2 FAIL /
18 unexecuted** of **14,073 selected**. All **116** historical b90 failure IDs
passed during that old epoch; it is not a green completed suite. The original
b90 failed full, initial guarded `119 PASS / 2 FAIL`, corrected `121 PASS` pair
and all older source-bound results remain verbatim below. Software polling and
cleanup are not a hard system memory quota, OOM diagnosis or general stability
guarantee.

The entire **B–F** goal, existing functionality, fixed supplied sources,
scientific FP64/low-precision workflows, current inventory/probe reconciliation
and opaque/metadata-free/manual obligations remain open. Ordinary SimpleLinear's
raw callback is still **FAILED**, with runtime/numerics **UNVERIFIED**. Held
shared validation and preparation namespace work is separate; no public
cutover, permanent selector, capability loss or full-model acceptance follows.

Current evidence: `root-pr277-ci-typing-final-staged-v1/manifest.json`;
`source-pr277-typing-final-primary-v1/receipt.json`;
`source-pr277-typing-final-secondary-v1/receipt.json`;
`source-pr277-scan-bf16-compat-final-v1-primary/receipt.json`;
`source-pr277-scan-bf16-compat-final-v1-secondary/receipt.json`;
`source-pr277-ci-typing-commit-v1-guard/receipt.json`;
`source-source-whole-entry-configuration-root-application-v1/source-inputs.json`;
`source-whole-entry-boundaries-v1-primary/receipt.json`;
`source-whole-entry-boundaries-v1-secondary/receipt.json`;
`root-whole-entry-paired-native-result-v1.json`.

### Retained d52 checkpoint record

The entire previous current record below is preserved verbatim at its original
source cut. This new section supersedes then-current and pending wording without
relabelling historical outcomes, criteria or receipts.

Snapshot: 2026-10-23. This unfinished LOCAL WIP follows **`e22ebcaa`**;
**`fe214590`** remains the last fully green checkpoint. The applied code cut for
this draft is **64ed** (**1,158 source pins**): the reviewed late SOURCE caller
configuration repair, CI/Gather corrections and LEGACY test-only expectation
correction are applied. No public
precision route or default has changed.

| Checkpoint gate | Actual source-bound result | Scope and next gate |
| --- | --- | --- |
| Ordinary full attempt, e22 epoch | **INTERRUPTED: 14,053 PASS / 2 FAIL / 18 unexecuted** of **14,073 selected** | All **116** historical b90 failure identities passed in this epoch. The failures and unexecuted cases remain; this is neither a completed green suite nor B–F acceptance. The earlier b90 **13,735 PASS / 116 FAIL** receipt remains historical, unchanged. |
| Initial guarded focused check, source `23df`, PRIMARY | **121 selected: 119 PASS / 2 FAIL**, **10.56 s**, source held; peak owned RSS **501,202,944 bytes**, no guard stop, process-group cleanup complete | Both failures are the new LEGACY False/True test's incorrect effective-x64 expectation. Production LEGACY remains unchanged; the test-only correction checks temporary global flag, overriding local/effective value and restoration. The corrected paired results are recorded below. |
| CI and Gather checkpoint corrections, source `23df` | Modern resolved-version guard expects **JAX/JAXLIB 0.12.0.1**; normal xdist invocation deliberately uses **one worker**; Gather asserts current **v7** and separately retains historical **v6** | Locks, support floors, original selectors, singleton/ordinal1 negatives, proof algorithms and raw criteria are unchanged. Static checks are not native-suite acceptance. |
| Corrected complete focused selection, source `64ed`, PRIMARY and SECONDARY | **121 PASS each**, same identities; PRIMARY **8.555 s**, SECONDARY **9.312 s**, zero failures/errors/skips, source held | Both runs completed cleanup with no guard stop; peak owned RSS **499,593,216 / 572,313,600 bytes**. This qualifies only the selected config/Gather checks, not the interrupted full suite or general resource safety. |

Strict docs, all seven normal hooks and exact staged-byte review remain required
checkpoint closeout gates. Their actual outcomes belong to the commit record;
the focused pair does not imply those later checks have passed.

The applied configuration helper snapshots global and local/unset values for
x64, matrix precision, dtype promotion and rank promotion, without installing a
local overlay. Its reviewed scope is the existing **late private SOURCE
conversion/model-callback boundary**. Earlier input/name/protocol evaluation and
the whole public-wrapper guarantee remain pending. LEGACY's temporary global
flag does not override an enclosing native local value. The test correction
retains the SOURCE-scope must-not-run guard and all six passing SOURCE/reentry
cases; it changes no production behavior.

The software runner guard observes owned-process resources and cleans up its
process group. Sampled limits and successful cleanup are not a hard cgroup/system
memory limit, an OOM/leak diagnosis, a bound on every single-case peak or a
machine-wide stability guarantee. The optional test cache/GC fixture, held public
validation draft and preparation namespace are excluded from this checkpoint.

Prior CNN, Scan, Gather, Eqx, admission and capture results retain their separate
source epochs below. Ordinary unchanged SimpleLinear still has a **FAILED raw
callback** with **UNVERIFIED runtime/numerics**. New focused observations do not
add closures to the frozen **3,704-caller** map, reconcile the changed Eqx recipe
provenance or close opaque/metadata-free/manual obligations. All existing
functionality, fixed supplied sources, original numerical criteria and full
**B–F** remain required through one coherent ordinary export experience. Public
behavior stays **LEGACY** and the ONNX IR floor stays **>=1.0.0**.

Current frozen checkpoint inputs: `root-ci-gather-pr-checkpoint-frozen-inputs-v1`;
`root-legacy-local-config-test-frozen-inputs-v1` (source `23df`, actual guarded
receipt `1649d872`);
`root-pr-config-gather-corrected-paired-result-v1.json`;
`source-pr-config-gather-corrected-checkpoint-v1-primary/receipt.json`;
`source-pr-config-gather-corrected-checkpoint-v1-secondary/receipt.json`;
`source-legacy-local-config-test-expectation-root-application-v1/source-inputs.json`; `source-source-caller-configuration-restoration-root-application-v1`;
`source-ci-gather-pr-checkpoint-sealed-v1`;
`source-legacy-local-config-test-expectation-sealed-v1`.

### Retained c875 follow-through record

The entire previous current record below is preserved verbatim at its original
source cut. The current checkpoint above supersedes its then-current wording;
no historical receipt or acceptance criterion is relabeled.

Snapshot: 2026-10-22. This unfinished LOCAL WIP continues
**`b74d8e937e7dc6557b788fcef6fc7dbf736cbaaf`**, whose normal local commit
completed all seven hooks and exact staged review with a clean worktree and no
push. **`fe214590`** remains the last fully green checkpoint. Results below keep
their own source epochs. The applied code cut at this snapshot is **c875**
(**1,157 input pins**). Native Array capture passes both complete selections;
ordinary SimpleLinear reaches its still-failed original graph callback. These
focused checks establish neither new full-suite nor migration acceptance.

| Follow-through gate | Actual source-bound result | Scope and remaining work |
| --- | --- | --- |
| CNN counterparts, source `645e`, PRIMARY and SECONDARY | **223 PASS each**, three complete owners; all **14** former failed identities PASS, source/Git held | Historical Add and current typed Gather correction proofs retain unchanged CNN sources, raw callbacks and numerical criteria. |
| Scan call metadata, source `172d`, PRIMARY and SECONDARY | **33 PASS each**; all **10** former failed identities PASS, source/Git held | Original physical carry/sliced-input, FP64 and packed-complex Function metadata remain strict. |
| Gather typed expectation, source `41a7`, PRIMARY and SECONDARY | **194 PASS each**; former ordinal1 failure PASS, source/Git held | Complete typed graph equality retains historical pins and strict native/runtime conjunction. |
| Requested-dtype Eqx factories, source `c7` | **30 PASS in each of four serial combinations**, PRIMARY/SECONDARY with outer x64 on/off: **8 original cases plus 22 construction guards**, source/Git held | Finite construction/generated-test qualification; not full SOURCE product or general numerical acceptance. |
| Historical Eqx typing, source `c7` | **FAILED**, two `no-any-return` errors in **572 checked files**; **RNG NOT RUN** | Later identity-only casts preserve the original native vmap results. This failed attempt is not relabeled. |
| Prior typing/RNG report, source `da58` | **PASS**, **572 source files** and **17 registered factory trace entries**, **48.941 s**, source/Git held | The 17 entries are reported RNG traces, not seventeen tests. Later code cuts require their own actual quality gates. |
| Admission/native dispatch, source `9ebd`, PRIMARY and SECONDARY | **556 PASS each**, identical complete four-owner selections; all **12 historical admission failure IDs PASS** on both stacks, source/Git held | Original custom-carrier exclusions, native frontend positives, Max18/26 structure and shared-B rank-five reshape remain strict. See the failed target-load evidence below. |
| Native Array capture, source `c875`, three complete owners | **525 PASS PRIMARY / 521 PASS SECONDARY**, source/Git held; all **17** former SECONDARY failed identities PASS on both stacks | The four PRIMARY-only AxisData controls reflect interpreter dataclass fields. Exact original-state capture is qualified; product/runtime/numerical acceptance is separate. |
| Ordinary unchanged SimpleLinear ordinal1, source `c875`, PRIMARY | **Two caller runs exit 0**, source/Git held; source-reference/schema/artifact/lowering **VALIDATED**, original raw callback **FAILED**, runtime/numerics **UNVERIFIED** | The callback expects `Gemm:12x3`; actual `Add`/`MatMul`/`Transpose` does not satisfy it. Original `None` criteria and callback remain unchanged; CLI exit is not product or numerical acceptance. |

The two current SimpleLinear observations retain FP32 inputs in both caller
scenarios and native/export outputs FP32 with x64 off, FP64 with x64 on. Native
eager/fresh-JIT calls and declared/concrete traces returned; all recorded state
checks report unchanged. They are separate current observations, not two additional frozen map
closures or accepted products. The original callback barrier remains work.

The admission pair records **12 actual CPU runtime-load observations** for the
three public creation opset27 cases, both CPU optimization modes and both stacks.
Each preserves its saved **opset27** artifact and exact structured **27 > 26**
target boundary: runtime evidence stays **FAILED**, numerical acceptance stays
**UNVERIFIED**, and `require_sample_acceptance` still raises. Passing test
assertions do not turn those target observations into numerical successes.
Prior capture source `7080` retains PRIMARY **514 PASS** and SECONDARY
**493 PASS / 17 FAIL** of **510**. Those failed receipts remain historical.
The earlier source `7080` PRIMARY combined selection remains **1,067 PASS /
3 FAIL** of **1,070**; its three new runtime27 test failures are retained as that
historical epoch. These scoped results do not rerun or subtract from the b90
ordinary full suite: **13,735 PASS / 116 FAIL / 0 ERROR / 0 SKIP** of **13,851**.
Its **54 actual model/state executions plus two guards** remain PASS under the
original source. All earlier failed receipts, callbacks and criteria stay visible.

The frozen **1,852-row / 3,704-caller** map retains **58 terminal mapping records
and 3,646 not-started cells**; mapping closure is not runtime or numerical
acceptance. The **63 opaque**, **104 metadata-free**, **three supplemental** and
**752 manual** obligations remain open. Seven Eqx raw rows now legitimately bind
deferred requested-dtype recipes with retained seeds, shapes and native call
order. Their callable provenance changed. Reconcile a fresh current native
inventory/probe explicitly; do not rewrite frozen **b875**, **ce87** or prior
**3,704-caller** outcomes, infer a current fingerprint, or assume an unchanged
discovery count from the safe projection digests.

Public options/defaults remain **LEGACY** and the ONNX IR floor remains
**>=1.0.0**. All existing functionality, fixed supplied sources, original owners,
numerical contracts and full **B–F** remain required through one ordinary
source-preserving export experience, without a permanent selector or capability
loss. Only demonstrated ONNX/target limits constrain support; implementation gaps
remain work. Public preparation namespace and coordinated cutover stay held.

This LOCAL WIP preserves reviewed unfinished progress. Relevant current quality
checks, strict docs/rendered links, all seven normal hooks and exact staged-byte
review are required closeout gates; their actual outcomes belong to the commit
record. They do not establish full-suite or migration acceptance.

Current receipts, relative to the external restart evidence directory:
`root-normal-followup-wip-commit-v1/receipt.json`;
`root-current-cnn-paired-qualification-identity-check-v1/receipt.json`;
`root-scan-source-call-metadata-paired-identity-check-v1/receipt.json`;
`root-current-gather-paired-identity-check-v1/receipt.json`;
`root-eqx-requested-dtype-four-way-identity-check-v1/receipt.json`;
`source-eqx-requested-dtype-current-typing-rng-v1/receipt.json`;
`source-eqx-vmap-typing-alias-current-typing-rng-v1/receipt.json`;
`root-runtime27-current-admission-paired-identity-check-v1/receipt.json`;
`root-secondary-native-array-binding-capture-safe-failures-v1/summary.json`;
`root-native-array-symbol-capture-paired-identity-check-v1/receipt.json`;
`root-current-simplelinear-ordinary-product-safe-summary-v1/summary.json`;
`root-current-simplelinear-next-callback-static-inputs-v1/inputs.json`
(`actual_callback_failure`);
`sol-eqx7-recipe-drift-static-accounting-v1/analysis.json`.

### Retained ce87-to-b74 follow-through record

The following earlier checkpoint cut is preserved verbatim with its original
source and then-current wording. The current section above supersedes its
pending/current claims; historical results and receipts are not relabeled.

Snapshot: 2026-10-22. The last **fully green** checkpoint remains
**`fe2145902d9231f21b9c867bf04d7fd2a9d0dddd`**. The latest stored **LOCAL WIP** is
**`ce87c2e090b71ed895cb806b2e627506e6ce266b`**: normal commit, strict docs, all
seven hooks, exact staged review and a clean worktree at commit. Its unfinished
status and retained failures are explicit; it is not a fully green milestone.

The prior qualified code cut is **`8603c9b588da755750099ae32c29b148396c1624fa80bd3793904f246a0e697e`**.
The formatted/rebound successor is **`2edc4a820273df105dae12309cad50c716350a49fe2f29ffeecd02f628109911`**;
its actual paired qualification is recorded separately below.
This intermediate checkpoint scope contains only ten reviewed follow-through
paths (five fixture/expectation, two Function-cache and three corpus-provenance
paths) plus these seven status documents. CNN and UnionType implementation are
outside this checkpoint. Existing criteria, raw callbacks, historical source
pins and approvals remain binding.

| Follow-through gate | Actual source-bound result | Scope |
| --- | --- | --- |
| Fixture/expectation five owners, c832 | **450 PASS each**, PRIMARY **64.459 s**, SECONDARY **75.608 s**, zero failures/errors/skips, source/Git held | All **28** exact formerly failed IDs are present and PASS in each selection. |
| Captured-state Function reuse, 094f | **310 PASS each**, PRIMARY **42.307 s**, SECONDARY **38.505 s**, zero failures/errors/skips, source/Git held | Complete captured-precision/context/NNX-state/native-capture owners; all **5** former failed IDs PASS, with original criteria and body/state distinctions retained. |
| Prior versioned Dot/Linear corpus, 8603 | **147 PASS each**, PRIMARY **17.351 s**, SECONDARY **18.819 s**, zero failures/errors/skips, source/Git held | Complete replay/numerics/leading-batch owners; all **38** former failed IDs PASS. Dot v3/Linear v2/control v7 retain historical descriptors and exact proof algorithms. |
| Prior typing/RNG, 8603 | **572 source files and 17 RNG factories PASS** in **57.396 s**, source/Git held | Run at 8603; earlier typing receipts keep their own epochs. |
| Formatted/rebound Dot/Linear corpus, 2edc | **147 PASS each**, PRIMARY **77.308 s**, SECONDARY **54.636 s**, zero failures/errors/skips, source/Git held | Fresh complete replay/numerics/leading-batch owners after normal Black formatting and the exact current descriptor rebind; prior 8603 receipts remain separate. |

The first follow-through normal-hook attempt exited **1** because Black
reformatted only the oracle; the other six hooks passed. Its AST stayed exact,
but whole-file descriptor provenance changed, requiring the current v7 literal
to be rebound from **add2fa9c** to **0ca2b0c8**. That failed hook attempt remains
history. Fresh strict-docs/seven-hook, staged-review and commit outcomes belong
to the normal commit body.

These selections retain separate source epochs and are not additive migration
coverage. The historical b90 ordinary full suite remains **13,735 PASS /
116 FAIL / 0 ERROR / 0 SKIP**; it has not been rerun. Its exact failed IDs and
bounded model/state positives remain below, without a new current full-suite
total or failure waiver. The complete **1,852-row / 3,704-caller** product map,
opaque/metadata-free/manual accounting, fixed supplied sources, numerical
contracts and full B–F/public transition remain open. Partial mapping outcomes
retain runtime/numerics **UNVERIFIED**. Public behavior stays **LEGACY**, and the
ONNX IR floor remains **>=1.0.0**.

The next intermediate LOCAL WIP preserves reviewed unfinished progress after
relevant actual checks, strict docs/rendered links, all seven normal hooks and
exact staged review. Its normal commit body records the actual closeout
outcomes. Those gates do not convert focused results into full-suite or
migration acceptance.

Current evidence: `source-ce87-followup-formatted-corpus-qualification-v1-primary/receipt.json`;
`source-ce87-followup-formatted-corpus-qualification-v1-secondary/receipt.json`;
`source-ce87-followup-all-seven-hooks-v1/receipt.json`;
`root-normal-local-wip-commit-v1/receipt.json`;
`source-local-wip-strict-docs-v1/receipt.json`;
`source-local-wip-all-seven-hooks-v2/receipt.json`;
`astra-final-local-wip-staged-review-v1/review.json`;
`source-ce87-fixture-expectation-paired-v1-primary/receipt.json`;
`source-ce87-fixture-expectation-paired-v1-secondary/receipt.json`;
`source-captured-state-function-reuse-qualification-v1-primary/receipt.json`;
`source-captured-state-function-reuse-qualification-v1-secondary/receipt.json`;
`source-current-dot-linear-counterpart-qualification-v1-primary/receipt.json`;
`source-current-dot-linear-counterpart-qualification-v1-secondary/receipt.json`;
`source-ce87-followup-current-typing-rng-v1/receipt.json`;
`root-ce87-followup-former-failures-qualified-v1/summary.json`.

### Retained pre-ce87 qualification cut

The following records the earlier source cuts and then-pending WIP gates.
The completed ce87 commit and later actual follow-through results above
supersede its present/future status wording; all evidence and failures remain
unchanged under their original sources.

Snapshot: 2026-10-22. The last completed local checkpoint remains
**`fe2145902d9231f21b9c867bf04d7fd2a9d0dddd`**: full PRIMARY **12,762 unique
cases**, SECONDARY and DIRECTMIN affected selections **919 cases each**,
production typing **559 files**, RNG **17 factories**, strict docs/rendered links,
all seven normal hooks and local commit with a clean worktree. Those gates
qualify that checkpoint alone. The later b90 full current-suite result is
**FAILED**, as recorded below. Unfinished LOCAL WIP preservation is distinct
from an accepted/green checkpoint and does not change that failure outcome.

The current code source at this draft cut is **`9134fcb6c2c574379205362055a5741b165b08400b45579097ceee513c21fd98`**
(**1,156 source/configuration pins**). It adds the independently approved
one-owner typing-only correction to b90: two local annotations and one local
key rename, with the complete prior engine AST recovered after those changes
are reversed. Fresh actual 9134 typing/RNG passes: **572 checked source files
and 17 RNG checks PASS** in **56.567 s**, source held. No full suite has been rerun at 9134; the actual failed full suite and
focused positive evidence below remain bound to **b90**.

At b90, the independently reviewed two-owner
integer-CSE live-branch restoration is applied. Its actual PRIMARY joint gate
records **331 PASS** in **188.562 s**, with zero failures/errors/skips and
source/Git held. Owner 197, including all 44 new cases, bridge 7, bindings 7, final
Constant 8, deployment 58, the full unchanged **Transformer 52** and permanent
MAIN repeated-Function/If 2 all pass. Both former explicit26 Transformer failure
IDs now pass under the same model recipes and criteria. Phase counts are
NOT_INSTRUMENTED. This closes the stated joint selection and Transformer owner
gate on PRIMARY; it does not complete all model, precision or migration work.

The same ordinary PRIMARY full suite completed **FAILED** at b90 in
**6,710.401 s**, exit 1, with source/HEAD/status held and all 1,156 source pins
exact: **13,851 unique cases = 13,735 PASS/116 FAIL/0 ERROR/0 SKIP**. The 116
failures span 23 owners; their exact identities and retained first traceback
locations are recorded once in the canonical terminal section below. Causes
remain **unclassified pending diagnosis/repair**; no generic waiver or
reclassification is applied. The 331 focused pass remains valid in its scope
and does not turn this failed full suite green.

All retained 56 selector dispositions are terminal and PASS: **54 actual model/state runs
plus 2 guard controls**, not 56 model runs. They comprise 44 original22
state cases, 10 native-binding model cases and 2 singleton criterion/guard tests.
The unchanged Transformer 52 and permanent MAIN If 2 selections also PASS in the
full suite. No case is dropped and no historical 19 passes are inherited. These
bounded positive gates do not accept the other 116 failures, complete every
numerical contract or finish B–F. The actual b90 typing gate **FAILED** with four
mypy errors in one engine file (572 checked source files); **RNG NOT RUN** after
the failed first command. The exact failure and subsequently applied minimal correction are
recorded below. The applied 9134 correction now passes fresh typing/RNG. Current
SECONDARY and DIRECTMIN selections each pass 279 cases; strict docs, all seven hooks and staged review
are required WIP-commit gates, with actual outcomes recorded in the commit body.

**Explicitly unfinished LOCAL WIP preservation** records reviewed unfinished
work; it is distinct from an accepted/green checkpoint, shipping or completion. It
may record unresolved real full-suite failures without repairing every failure
first. The failed terminal outcome is recorded and the relevant current typing/RNG
and paired selections pass. Strict docs/rendered links, all seven normal hooks
and final staged-byte review remain required commit gates. The applied
typing-only correction has its own actual source-bound qualification. Four
fixture candidates and one expectation candidate remain external/unapplied and
are outside this WIP source cut. The normal commit body records actual relevant
checks, strict docs, seven-hook and staged-review outcomes for the preservation
commit; it does not assert green full-suite acceptance. An accepted checkpoint
and full B–F completion still require their complete actual gates.

Earlier e6 287=285 PASS/2 FAIL, source51 275=273 PASS/2 FAIL, a8
255=253 PASS/2 FAIL and other receipts retain their exact sources and failure
boundaries below. They do not qualify the new production bytes. The dd1
permanent repeated-Function/If owner adds only its path-marker header; its
body/AST remain the externally qualified **287414f7** version. The user’s
Poetry update is retained: modern JAX/JAXLIB **0.12.0.1** and ONNX IR **1.1.0**.
The separate uv lock is unchanged and the declared ONNX IR minimum
**>=1.0.0 is retained**. PRIMARY’s updated packages form a distinct environment
epoch. **Earlier source/stack numerical and full-suite results do not qualify
that updated environment.**

Shared work preserves original callable
and owner preparation, actual NNX method/state binding, native numeric carriers,
rank-four PRNG and additional rank/axis/Tile/Split/Concat layouts. The positive
symbolic-cast continuation preserves actual cast rounding and named Function
boundaries. Native TopK adds FP16, full UINT64 and the actual unstable option
to the previously tested integer/FP32/FP64 contracts; typed empty results retain
or remove the actual batch symbol as specified. BF16 uses the same signed
16-bit word adapter and original typed values at explicit26. Its portable
NaN-word CPU semantics and strict compiled-native equivalence have separate
results: a passing portable test never accepts a failed native comparison.
Requested targets and original model recipes remain unchanged.

The audited NaN-free and empty BF16 cases match all three native reference
modes and both CPU settings. Portable original-word TopK also passes the NaN
fixtures; their strict native comparisons remain **FAILED** on both JAX stacks.
PRIMARY has value-word differences; SECONDARY also has index differences.

The private HIGHEST route still requires its declared CPU runtime/build/options
and actual optimized graph correspondence. Exact lexical ownership, repeated
call bindings, standard shape data propagation and direct Constant-definition
origins now pass the unchanged repeated-Function/If regression on PRIMARY,
SECONDARY and DIRECTMIN, with both CPU profiles and all three native references.
These results qualify that recipe and the directed owner selection only.

The applied b90 proof covers the exact live integer-expression branches and
their downstream Gather-index and floating-Where-mask endpoint partitions.
Only proved current-target dependency closures are cloned and proved endpoint
ports redirected. Actual original Cast bytes and artifacts, Gather/floating
Where data bindings, the protected bridge, mathematical rules, numerical
criteria and final precision tuple equality remain unchanged. This is a
bounded live-branch restoration, not blanket CSE acceptance or a relaxed final
comparison. The new 44-case controls and the unchanged Transformer 52 owner pass
in the actual 331-case PRIMARY gate.

The earlier e6 final delta remains retained: all 17 matrix tuples matched,
Cast totals 24/32 were exact, and only two INT64-to-INT32 direct-consumer sets
differed 12-to-3 each. Its repeated negative-index Less/Add/Where metadata was
diagnostic and did not itself prove downstream redirects. The subsequent
independently reviewed live-branch proof and actual b90 owner results resolve
those two selected Transformer failures prospectively. The old e6 failed run
remains 285 PASS/2 FAIL in its own epoch.

Correction to ROOT’s earlier diagnostic interpretation: inference calls were
misnumbered because `precision.contract(original)` performs the first inference
before the bridge’s four initial calls. Bridge pre-original validation is
call 2, pre-optimized validation call 3, source-inline inference call 4, and
**INITIAL TARGET inline inference is call 5**. The earlier ROOT statement that
the initial inferred target had zero symbols was **wrong**. Raw original and
raw optimized models have no explicit symbols, but call 5 creates six inferred
symbols. The retained a8 transition observer finds all six already protected at
the **first refresh callback**, with an empty active fresh-symbol ledger; the
selected symbol has 29 original/30 prepared occurrences. No callback admits an
active fresh generation. This corrects the attribution, not any sealed receipt
or historical test result. The earlier observer’s selected refinement to 3 did
not establish the returned value of every former occurrence.
At a8 the actual private type-merge conflict remained F32[generated symbol,4]
versus F32[3,4]; that failure retains its own source-bound receipts.

That initial-provenance bootstrap is now independently approved and applied at
source51. The corrected a8 diagnosis and all sealed receipts remain historical
evidence. The source51 Gemm failure and then-planned all-FMM selection remain historical;
e6 applied that selection and reached the later final comparison; b90’s bounded
live-branch proof now passes that owner gate. These steps
do not relax original symbols, whole-consistent bindings, actual occurrences
or other proof gates. No symbol-prefix rule, source-type seed, metadata clearing
or Cast/default relaxation is accepted.

Each row below retains its own source epoch and scope. Counts overlap and **must
not be added into coverage, a full-suite total or all-model acceptance**.

| Gate and source scope | Actual result | Qualification and remaining work |
| --- | --- | --- |
| Committed wrapper checkpoint, fe214590 | PRIMARY 12,762; SECONDARY 919; DIRECTMIN 919; typing 559/RNG 17; strict docs, seven hooks and local commit complete | Historical completed source only. |
| Prior green three-stack directed owner gate, c5 source | PRIMARY **158 PASS** in 101.640 s; SECONDARY **158 PASS** in 40.113 s; DIRECTMIN **158 PASS** in 32.701 s; zero failures/errors/skips and source/Git held on all three | Same selection: final Constant origins 8, engine 83, lexical bindings 7, deployment 58 and unchanged external repeated-Function/If 2. Both external CPU modes preserve all three native bitwise references. Phase counts are NOT_INSTRUMENTED. These are scoped results, not current full-suite/all-model acceptance. |
| Prior production typing/RNG, c5 source | **572 production files and 17 RNG factories PASS**, source/Git held, in71.369 s | This earlier receipt does not qualify the later 014 production refresh. Its later f82 typing/RNG qualification is recorded separately. |
| Prior production typing/RNG, f82 source | **572 production files and 17 RNG factories PASS**, source/Git held, in **73.220 s** | Fresh actual f82 qualification after the generic refresh and test-only fixture correction; it does not qualify the later a8 production changes. The log digest matches the earlier c5 log, but this is a separate source-bound run/receipt. Current full suite, all 56 selectors, final docs/hooks and new checkpoint remain pending. |
| Prior production typing/RNG, a8 source | **572 production files and 17 RNG factories PASS** in **56.291 s**, source/Git held | A fresh actual a8 run/receipt qualifies these production bytes. Its log digest matches earlier c5/f82 logs; their source-bound receipts remain separate. The later source51 bootstrap has its own fresh typing/RNG receipt below. Full suite, all 56 selectors, final docs/hooks and new checkpoint remain pending. |
| Prior production typing/RNG, source51 | **572 production files and 17 RNG factories PASS** in **66.079 s**, source/Git held | Fresh actual source51 qualification after the applied initial-provenance bootstrap. Earlier a8/f82/c5 receipts keep their own sources even though their log digests match. The later applied e6 selection has its own fresh typing/RNG receipt below; full suite/final gates remain open. |
| Actual current typing attempt, b90 source | **FAIL**: four mypy errors in `_deployment_fusions.py` at 2452,2971,2973,2986; 572 source files checked; **RNG NOT RUN**; source held, **96.369 s** | Errors are tuple-assignment type, reused key type/unreachable branch, and missing local nodes annotation. The exact minimal annotation/local-name correction is now independently approved and applied at 9134; fresh 9134 typing/RNG now PASS in 56.567 s. This failed b90 attempt and RNG-not-run stay recorded. |
| Prior production typing/RNG, e6 source | **572 production files and 17 RNG factories PASS** in **56.147 s**, source/Git held | Fresh actual e6 qualification after applied unit-FMM selection. Same-digest earlier logs retain their distinct source/time receipts. The later b90 four-error typing attempt and applied 9134 fresh typing/RNG pass are recorded separately below. Complete model/full-suite and migration gates remain open. |
| Generic refresh actual gate, 014 PRIMARY source | **232 PASS / 3 FAIL** of 235 unique cases in **63.313 s**, zero errors/skips and source/Git held | Engine 101=100 PASS/1 new fixture field-presence FAIL; the other 17 new engine cases pass. Bridge 7, bindings 7, Constant 8, deployment 58 and permanent MAIN If 2 all PASS. Transformer 52=50 PASS/2 FAIL at the private type merge of a generated inferred dimension against known 3 in both26 orders. Phase counts are NOT_INSTRUMENTED. The later fixture correction does not relabel this failed gate. |
| Retained six-owner directed gate after test-only fixture packaging, f82 source | **183 PASS on each stack**: PRIMARY **34.320 s**, SECONDARY **26.671 s**, DIRECTMIN **24.450 s**; zero failures/errors/skips and source/Git held on all three | Engine 101, bridge 7, bindings 7, Constant 8, deployment 58 and permanent MAIN If 2 all pass. Phase counts are NOT_INSTRUMENTED. The only later change makes empty domain/overload fields explicit in fixture construction; full Cast byte assertions, production and numerical criteria stay exact. Transformer 52 is outside this selection and retains its 50 PASS/2 FAIL result at 014. Current-production typing/full-suite gates remain separate. |
| Prior fresh-symbol joint gate, a8 PRIMARY source | **253 PASS / 2 FAIL** of 255 unique cases in **71.234 s**, zero errors/skips and source/Git held | Engine 121, including all 20 new cases, bridge 7, bindings 7, Constant 8, deployment 58 and permanent MAIN If 2 all PASS. Transformer 52 remains 50 PASS/2 FAIL in both explicit26 orders. Phase counts are NOT_INSTRUMENTED. In this a8 epoch the refresh-only tracker did not address the initial-symbol barrier. The later source51 bootstrap has its own changed failure boundary below. |
| Prior six-owner fresh-symbol gates, a8 SECONDARY and DIRECTMIN sources | **203 PASS each**: SECONDARY **20.742 s**, DIRECTMIN **17.837 s**; zero failures/errors/skips and source/Git held | Same six passing owners as the current PRIMARY gate, excluding Transformer 52. Positive IR1.0 evidence remains scoped to this selection; the IR floor is retained. |
| Prior initial-provenance joint gate, source51 PRIMARY | **273 PASS / 2 FAIL** of 275 unique cases in **90.392 s**, zero errors/skips and source/Git held | Owner 141, including 20 new initial-provenance cases, bridge 7, bindings 7, Constant 8, deployment 58 and permanent MAIN If 2 all PASS. Transformer 52 remains 50 PASS/2 FAIL; the initial-dimension barrier is overcome and both explicit26 cases now fail at later Gemm correspondence. Phase counts are NOT_INSTRUMENTED. Current paired 223 and typing 572/RNG17 pass separately below; prior a8 paired 203 retains its own epoch. |
| Prior six-owner initial-provenance gates, source51 SECONDARY and DIRECTMIN | **223 PASS each**: SECONDARY **24.701 s**, DIRECTMIN **39.205 s**; zero failures/errors/skips and source/Git held | Owner 141, bridge 7, bindings 7, Constant 8, deployment 58 and permanent MAIN If 2 pass. Transformer 52 is outside these selections and retains its current PRIMARY 50 PASS/2 FAIL result. Positive IR1.0 evidence remains scoped; the floor stays >=1.0.0. |
| Prior unit-FMM joint gate, e6 PRIMARY | **285 PASS / 2 FAIL** of 287 unique cases in **56.169 s**, zero errors/skips and source/Git held | Owner 153 including 12 new cases, bridge 7, bindings 7, Constant 8, deployment 58 and permanent MAIN If 2 all PASS. Transformer 52 remains 50 PASS/2 FAIL at final bound dataflow; previous unit-FMM/Gemm barriers pass. All17 matrix tuples match; only two INT64-to-INT32 Cast consumer sets differ 12-to-3. Phase counts are NOT_INSTRUMENTED. At e6 exact CSE/downstream redirect proof remained pending. Its later bounded b90 resolution and actual 331 PASS are separate; this old failed run is not relabeled. |
| Prior six-owner unit-FMM gates, e6 SECONDARY and DIRECTMIN | **235 PASS each**: SECONDARY **16.862 s**, DIRECTMIN **15.595 s**; zero failures/errors/skips and source/Git held | Same six passing owners as current PRIMARY, excluding Transformer 52. Positive IR1.0 evidence remains scoped; prior source51 paired 223 and a8 paired 203 keep their own epochs. |
| Current live-branch joint gate, b90 PRIMARY | **331 PASS** in **188.562 s**, zero failures/errors/skips and source/Git held | Owner 197 including 44 new cases, bridge 7, bindings 7, Constant 8, deployment 58, full Transformer 52 and permanent MAIN If 2 all PASS. The former two Transformer 26 failure IDs pass under unchanged recipes/criteria. Phase counts are NOT_INSTRUMENTED. This scoped owner gate does not complete the full suite, all models, B–F or public cutover. |
| Current ordinary PRIMARY full suite, b90 source | **13,735 PASS/116 FAIL/0 ERROR/0 SKIP** of 13,851 unique cases; **FAILED**, exit 1, in **6,710.401 s**; source/HEAD/status held and source pins exact | All 54 actual model/state runs and both singleton guards PASS:56 selected checks. Transformer 52 and MAIN If 2 also PASS. The 116 failures/23 owners remain unclassified and unwaived; exact IDs are recorded once below. Focused331 PASS is separate. No duplicate broad full or56-selector rerun is required merely to preserve reviewed LOCAL WIP; accepted checkpoint/full migration require their complete gates. |
| Applied typing-only correction, current 9134 code | One reviewed engine owner applied; **572 source files and 17 RNG checks PASS** in **56.567 s**, source held | Two local annotations and one local key rename recover the full old engine AST. No full-suite rerun is implied: b90 full failure and focused passes keep their original source scopes. Fresh SECONDARY and DIRECTMIN each pass 279 cases below; strict docs/hooks/staged review are required preservation-commit gates. |
| LOCAL WIP preservation boundary | **Explicitly unfinished; not accepted/green checkpoint** | Relevant actual checks, terminal failure accounting, all seven normal hooks and final staged-byte review are required. Unresolved full-suite failures can remain honestly recorded; green/accepted checkpoint and full migration gates stay separate. Four fixture candidates and one expectation candidate remain external/unapplied; public preparation namespace/cutover remain held. |
| Fresh current 9134 six-owner gates, SECONDARY IR1.0 and DIRECTMIN | **279 PASS each**: SECONDARY **17.963 s**, DIRECTMIN **15.867 s**; zero failures/errors/skips and source/Git held | Engine 197, bridge 7, bindings 7, Constant 8, deployment 58 and permanent MAIN If 2 pass on both stacks. DIRECTMIN retains all 15 direct dependency floors. Transformer 52 and the ordinary full suite were not rerun at 9134; their b90 receipts remain separate. |
| Retained paired BF16/alias/deployment selection, 78ed source | **200 PASS**, zero failures/errors/skips in **34.70 s**: complete TopK **99**, Function aliases **9**, deployment fusions **34** and deployment targets **58** | Actual direct BF16 GatherElements and both CPU profiles pass their portable source-word checks. The two NaN portable test IDs are separate from strict native-equivalence acceptance; actual per-role/profile outcomes are retained below. The same SECONDARY200 selection also passes in27.81s at78ed, with source/Git held on both stacks. The later a893 typing/RNG result remains in its own epoch; prior e6 typing is listed above. Complete-model qualification remains pending. Phase counts were not instrumented. |
| Retained BF16 exact-equivalence rollups, 78ed source | Both stacks: audited NaN-free/empty **22 validated observations per native role**, zero changed value/index words. NaN portable observations **12 PRIMARY / 8 SECONDARY PASS**; strict eager/fresh/warmed comparisons each retain **12 PRIMARY / 8 SECONDARY FAILED** | PRIMARY retains **1,386 changed value words / zero changed index words**; SECONDARY retains **924 changed value words / 756 changed index words**. The counts differ because PRIMARY also tests the actual available is_stable=False route. Portable original-word success never accepts native-equivalence FAILED; no general BF16 bitwise-native claim. These are comparison observations, not a coverage total. |
| Existing DIRECTMIN diagnostic selection, 78ed source | The same **200 PASS**, zero failures/errors/skips, in27.97s with source/Git held | ONNX1.20.1 lacks BitCast26: positive26 cases retain the required actual representation-unavailable counterpart. This is **not positive BF16/ORT26 parity at the old floor**. Applicable empty23, integer23 and Function-alias paths pass; actual positive26 evidence belongs to PRIMARY/SECONDARY above. |
| Paired FP16/UINT64/unstable widths, 6a source | PRIMARY complete79 TopK plus four sensitive matrix cases **83 PASS**; SECONDARY six-owner selection **395 PASS**, including all79 TopK | Genuine named/JIT/If scopes pass. Initial combined453 failure runs remain historical; no arithmetic combination relabels them as453 PASS. |
| Empty-result discovery and shared Function alias correction | cf2b PRIMARY **90 PASS / 1 FAIL** of91; the failure is duplicate Function outputs after no-op INT32 Cast removal. Later12d PRIMARY alias9 and unchanged TopK91 pass within **242 PASS / 2 FAIL** of244 | Final same-graph typed Identity aliases preserve output slots after optimization. The two12d failures are original Transformer shape-metadata obligations below; old91 is not relabeled by composition. |
| Shared owner/state and transport gates, prior pinned sources | Owner/state PRIMARY 391 and SECONDARY 391 PASS; transport PRIMARY/SECONDARY/DIRECTMIN 216 each PASS; b46 DIRECTMIN affected 506 PASS | Retained focused evidence; no current full qualification. |
| Integer transport, original d430/e795 source epochs | PRIMARY 376 PASS; SECONDARY 521 PASS; DIRECTMIN **514 PASS / 7 FAIL** of 521 | Four actual FP16/BF16 Range27 schema failures and three actual explicit26 ORT load-ceiling failures remain historical FAILED results. No version-based skip, target downgrade or support-floor change. |
| Positive symbolic cast and existing owners, 67cb source | PRIMARY **571 PASS**, zero failures/errors/skips | Original typed-boundary, nonleading-layout and tensor/index owners included; separate original-model acceptance below. |
| Original owner 7, new x64-on acceptance at ae1 source | **1 PASS**: unchanged six-layer VisionTransformer, B=2/5 and TFFT state/RNG sequence, three native references, one artifact, both CPU profiles and **32 target runs** | Closes this actual new cell only. The earlier six-cell observation remains 5 PASS / 1 FAIL. |
| Original nested owners 7/8/9, 279 source | Terminal XML records **5 PASS / 1 FAIL**; 7-off has 32 runs, 8-off/on 16 each and 9-off/on 32 each; each passing cell has one artifact and both CPU profiles | Original driver receipt is absent and process exit code **UNOBSERVED**. The 7-on cast-layout failure is retained alongside its later separate pass. |
| Native TopK, c97 PRIMARY and dcf SECONDARY | All **55 new cases PASS on both stacks**; the PRIMARY typed-builder 12 and legacy TopK 4 also pass | These55 cover their then-admitted widths with native/artifact/CPU evidence. Later width/empty/BF16 gates are separate; broader shapes/options and complete original models remain required. |
| Explicit26 Min/Max/clamp, dcf PRIMARY and SECONDARY | All **56 new and 28 old cases PASS on both stacks**; SECONDARY TopK+Min/Max selection **139 PASS** | Earlier 15 fixture failures remain recorded; no relaxed numerical comparisons. |
| Activation/Min/Max combined gate, dcf PRIMARY | **257 PASS / 4 FAIL** of 261; all 56 new and 28 old Min/Max cases pass | The four activation routes pass disabled optimization and fail ALL with exactly ten signed-zero word differences each and no other differences. The later shared selection repair is applied; current directed deployment results qualify their stated scopes only. This historical failure and criterion remain intact. |
| Original Eqx owner 18 and default-anchor controls | PRIMARY **22 PASS** at 666 source; SECONDARY **22 PASS** at dcf source: 20 directed controls plus both original caller-x64 cells per stack | Each original cell retains one artifact, both CPU profiles and **16 target runs**, including exact directed masks. Earlier opaque-default failures remain retained. |
| Original Linen 19 and SpectralNorm 20, retained PRIMARY gate | Both x64 cells for each pass; Linen has **32 runs per cell**, SpectralNorm **8 per cell**, with one artifact and both CPU profiles | The original six-case gate is still **4 PASS / 2 FAIL** because Eqx18 failed before native/artifact evidence in that epoch. The later Eqx gate is separate. |
| Full original GPT owner 0, 6825 PRIMARY and dcf SECONDARY sources | Both raw-x64 cells **PASS on each stack**; each cell retains one artifact, both CPU profiles, **8 target runs** and **24 comparisons** to three native roles under frozen criterion **67cd821a** | The actual raw callback executes and still returns **False**; raw criteria=None and stored UNVERIFIED fields are unchanged. Exact native-unused proof covers five Function formals across 15 calls/14 definitions with no root exemption. These are prior-environment results, not updated ONNX IR 1.1 qualification. |
| Repeated same Function containing If, c5 external regression and dd1 MAIN packaging | Both CPU-profile cases **PASS on PRIMARY, SECONDARY and DIRECTMIN**, including all three native bitwise reference comparisons under the unchanged original criteria | The permanent MAIN owner adds only the path marker: marked post **c44ff518**, exact semantic body/AST **287414f7**. Historical a893 **2 FAIL during export**, before any returned model or CPU session, and later scope/Gemm/final-origin failures remain retained. This closes the actual reproducer; general model/state/scope obligations and the fresh suite remain open. |
| Original/prepared 52-case Transformer owner, current b90 PRIMARY gate | **52 PASS**, including both former explicit26 failure IDs, within the actual 331-case joint gate | Exact live-branch restoration proves Gather-index and floating-Where-mask partitions and actual downstream redirects. Original Cast bytes/artifacts, frozen **d2ecaed3**, source recipes, mathematical helpers and final precision equality remain unchanged. Prior e6/51/a8/014 failures remain their own epochs. This closes the current owner gate only; full-suite/model inventory and complete migration obligations remain open. |
| Original vision Transformer 14, retained b46 PRIMARY source | x64-off cell PASS: one full artifact, both CPU profiles and **32 target runs** | Other caller/stack cells are separate. Internal masks remain UNVERIFIED where the unchanged return contract exposes no mask tape. |
| Prior updated-environment typing/RNG, 84e source | **569 production files and 17 RNG factories PASS**, source/Git held | Fresh qualification with JAX/JAXLIB0.12.0.1 and ONNX IR1.1.0. The prior dcf568/RNG17 pass and initial c6 import-untyped failure remain separate history; full suite, final precision docs/seven hooks and new commit remain open. |
| Later typing/RNG, d500 source | **571 production files and 17 RNG factories PASS**, source/Git held | Includes the applied zero/width/fusion typing correction. 78ed typing later fails on one redundant cast; RNG is not run. This earlier pass does not qualify later bytes. |
| Retained typing attempt, 78ed source | **FAIL**: one redundant typing.cast at source_top_k.py95 in571 production files; **RNG NOT RUN** | The later typing-only identity/import removal is applied; fresh actual a893 typing/RNG passes below. The78ed failed attempt and RNG-not-run remain retained. |
| Prior production typing/RNG, a893 source | **571 production files and 17 RNG factories PASS**, source/Git held, in55.33s | The only continuation from78ed removes standard typing.cast identity and its unused import. Prior runtime receipts retain their78ed scope; no numerical rerun is required solely for that typing cleanup. Full current suite, docs, final hooks and new commit remain open. |
| Updated PRIMARY core and capability controls, c6ea source | **387 PASS** with JAX/JAXLIB **0.12.0.1** and ONNX IR **1.1.0**, zero failures/errors/skips | IR builder/Function, optimizer, facade, capability/evidence and IR-only policies pass in this focused selection. This is fresh updated-environment evidence, not a full suite or all-model result. Matched DIRECTMIN core+original521 gate records **907 PASS / 1 FAIL** of 908; the remaining failure is the unchanged facade BitCast26 case with no installed ONNX schema, not an observed ONNX IR API failure. The later exact test-only correction passes its two-owner157-case selection on PRIMARY and DIRECTMIN at84e. That closes the specific counterpart prospectively; the old908 result stays907 PASS / 1 FAIL. |
| General installed schema/runtime diagnostics | Reviewed production helper applied at5d; PRIMARY387 and SECONDARY diagnostic47+tensor/index296 **343 PASS** atc6ea; later PRIMARY and DIRECTMIN facade/diagnostic selections **157 PASS each** at84e | The seven original DIRECTMIN IDs reach precise actual failed-capability counterparts; schema/load remain FAILED and numerics UNVERIFIED. The facade uses one unchanged actual attempt: capable exports retain all original report assertions; the old ONNX stack records its exact missing BitCast26 schema without fabricating a returned artifact/report. No historical result is relabeled. |
| Updated dependency documentation, c99 source | Actual strict MkDocs **PASS** in **11.038 s**, source/Git held, for the 22 Oct dependency page with updated packages | The earlier c6ea10.15 s pass remains its own epoch. Both builds qualify their applied dependency page and prior documentation only; this eight-document current-status refresh and final checkpoint gates remain separate. |
| Remaining-width TopK native observations, separate actual epochs | SECONDARY v1: **240 cells / 720 numerical calls**, all return, zero numerical-call exceptions, but **240 trace-metadata failures**. Updated PRIMARY v2: **480 cells / 1,440 numerical calls**, all return, zero numerical-call exceptions, but **480 trace-metadata failures** | Both native context classes declare an uninitialized configs slot; these trace snapshots failed independently of completed numerical calls. A later trace-only supplement has separate successful observations below. BF16 word/index mismatches and PRIMARY v1's zero-call version rejection remain retained; numerical-return counts alone never qualify SOURCE export/runtime or trace metadata. |
| BF16 retained-return diagnostic | Actual post-observation analysis finds NaN-word canonicalization with sign preserved; finite, zero, subnormal and infinity words are unchanged | C(source[actual index]) explains returned words on both stacks. Canonicalizing before sort explains the older stack's index differences; original-before-sort explains the updated primary's both stability modes. These are explanatory hypotheses over retained data, not a changed fixture/comparator or accepted lowering. Later actual shape-only HLO observations are recorded separately below; no native mismatch or comparator is changed. |
| Actual trace-only supplement, 84e source | PRIMARY **480** and SECONDARY **240 successful trace snapshots**; each54 preparations include nine UINT64 raw-x64-false cases; **zero numerical TopK calls** | The separate45 admitted preparation-parity checks pass per stack. Original480/240 trace failures remain historical. This supplies typed trace metadata, not numeric/export/runtime acceptance. |
| Actual shape-only BF16 HLO, 84e source | PRIMARY **8** and SECONDARY **4 successful compilations**, **zero executions**; original exit2/incomplete-extraction status retained | PRIMARY sorts one packed s32 key; SECONDARY sorts F32+iota with comparator conversion through BF16/s16 words. ROOT classifies228 extraction gaps across24 required artifacts as debug tables or exact empty sdy.mesh, with no unclassified lines. Opaque backend semantics remain unclaimed; compiler evidence is separate from numerical/export acceptance. |
| Updated IR return modes | All **four tests PASS on PRIMARY IR1.1** and all **four PASS on DIRECTMIN IR1.0** | Same return-mode scope, separate actual receipts. The declared IR minimum **>=1.0.0 is retained**; these four IDs do not establish a complete minimum-stack suite. |
| Initial updated-source typing attempt, c6ea | **FAIL** on one untyped import in the new capability helper; RNG was not run after the failed typing command | A scoped typing-only correction was applied. The later fresh84e569/RNG17 PASS is separate; this failed attempt and the earlier dcf568/RNG17 pass are retained. |

### Actual terminal full-suite failures at b90

The **116 failed identities in 23 owners** below come from the exact terminal
safe v2 summary (SHA256 `64fb92be4388b7901ca8ca585f21ff9c8e94bb2d60a052ad3e725d3b79d514cd`).
It corrects only the rendering of eight LEGACY class selectors to the actual
`module.py::Class::test` identities. All 116 IDs match the actual failed progress
set. The v1 owner/test pairs, statuses and trace hashes were correct and remain
preserved; its malformed class paths are not used as the canonical IDs.
It retains each failure’s trace hash and first traceback path/line without
arrays or model payloads. Owner grouping is accounting, not cause classification.
Every listed cause remains **unclassified; diagnosis/repair required**. The
completed full run remains FAILED; none is waived by the 54+2 selected passes,
the 331 focused gate or the LOCAL WIP preservation decision.

| Failed owner | Failed identities |
| --- | ---: |
| `tests.extra_tests.converter.test_source_where_dispatch` | 1 |
| `tests.extra_tests.converter.test_source_elementary_reductions` | 1 |
| `tests.extra_tests.converter.test_source_loop_admission` | 7 |
| `tests.extra_tests.converter.test_source_square_variance` | 2 |
| `tests.extra_tests.converter.test_source_initializers` | 8 |
| `tests.extra_tests.converter.test_source_array_creation_promotion` | 1 |
| `tests.extra_tests.converter.test_source_native_conv` | 4 |
| `tests.extra_tests.converter.test_source_jnp_binary_dispatch` | 2 |
| `tests.extra_tests.converter.test_source_precision_corpus_replay` | 21 |
| `tests.extra_tests.converter.test_source_finalization` | 12 |
| `tests.extra_tests.converter.test_source_profile` | 2 |
| `tests.extra_tests.converter.test_source_precision_corpus_numerics` | 6 |
| `tests.extra_tests.converter.test_source_native_cnn_nonzero` | 8 |
| `tests.extra_tests.converter.test_source_native_cnn_pool` | 2 |
| `tests.extra_tests.converter.test_source_leading_batch_corpus` | 11 |
| `tests.extra_tests.converter.test_source_fixed_parameter_models` | 5 |
| `tests.extra_tests.converter.test_source_scan_physical_metadata` | 10 |
| `tests.extra_tests.converter.test_source_captured_precision` | 5 |
| `tests.primitives.test_eqx.Test_weight_norm` | 3 |
| `tests.primitives.test_eqx.Test_conv` | 2 |
| `tests.primitives.test_eqx.Test_gru_cell` | 1 |
| `tests.primitives.test_eqx.Test_lstm_cell` | 1 |
| `tests.primitives.test_eqx.Test_spectral_norm` | 1 |

<details>
<summary>All 116 exact failed selector identities</summary>

```text
tests/extra_tests/converter/test_source_where_dispatch.py::test_boolean_to_float_accepts_original_explicit_conversion
tests/extra_tests/converter/test_source_elementary_reductions.py::test_original_gather_trigonometric_anchor_preserves_raw_criterion_and_both_native_roles[1]
tests/extra_tests/converter/test_source_loop_admission.py::test_zero_fill_admission_keeps_other_opset_routes_excluded[16]
tests/extra_tests/converter/test_source_loop_admission.py::test_zero_fill_admission_keeps_other_opset_routes_excluded[18]
tests/extra_tests/converter/test_source_loop_admission.py::test_zero_fill_admission_keeps_other_opset_routes_excluded[21]
tests/extra_tests/converter/test_source_loop_admission.py::test_zero_fill_admission_keeps_other_opset_routes_excluded[24]
tests/extra_tests/converter/test_source_loop_admission.py::test_zero_fill_admission_keeps_other_opset_routes_excluded[27]
tests/extra_tests/converter/test_source_loop_admission.py::test_zero_fill_admission_keeps_other_creation_wrappers_excluded[ones]
tests/extra_tests/converter/test_source_loop_admission.py::test_zero_fill_admission_keeps_other_creation_wrappers_excluded[full]
tests/extra_tests/converter/test_source_square_variance.py::test_max_never_falls_through_to_unaudited_legacy_route[float32-26]
tests/extra_tests/converter/test_source_square_variance.py::test_symbolic_singleton_reshape_retains_transform_exclusions[rank-five]
tests/extra_tests/converter/test_source_initializers.py::test_source_relu6_typed_initializers_survive_plugin_and_function_export[inline-float16]
tests/extra_tests/converter/test_source_initializers.py::test_source_relu6_typed_initializers_survive_plugin_and_function_export[inline-bfloat16]
tests/extra_tests/converter/test_source_initializers.py::test_source_relu6_typed_initializers_survive_plugin_and_function_export[inline-float32]
tests/extra_tests/converter/test_source_initializers.py::test_source_relu6_typed_initializers_survive_plugin_and_function_export[inline-float64]
tests/extra_tests/converter/test_source_initializers.py::test_source_relu6_typed_initializers_survive_plugin_and_function_export[nested-float16]
tests/extra_tests/converter/test_source_initializers.py::test_source_relu6_typed_initializers_survive_plugin_and_function_export[nested-bfloat16]
tests/extra_tests/converter/test_source_initializers.py::test_source_relu6_typed_initializers_survive_plugin_and_function_export[nested-float32]
tests/extra_tests/converter/test_source_initializers.py::test_source_relu6_typed_initializers_survive_plugin_and_function_export[nested-float64]
tests/extra_tests/converter/test_source_array_creation_promotion.py::test_creation_repairs_do_not_widen_product_admission[full]
tests/extra_tests/converter/test_source_native_conv.py::test_unchanged_complete_cnn_exports_source_with_separate_raw_callback[False-0]
tests/extra_tests/converter/test_source_native_conv.py::test_unchanged_complete_cnn_exports_source_with_separate_raw_callback[False-1]
tests/extra_tests/converter/test_source_native_conv.py::test_unchanged_complete_cnn_exports_source_with_separate_raw_callback[True-0]
tests/extra_tests/converter/test_source_native_conv.py::test_unchanged_complete_cnn_exports_source_with_separate_raw_callback[True-1]
tests/extra_tests/converter/test_source_jnp_binary_dispatch.py::test_nonselected_binder_and_existing_profile_exclusions_remain
tests/extra_tests/converter/test_source_jnp_binary_dispatch.py::test_where_full_adapters_and_dynamic_arange_compatibility_remain
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure[False-0]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure[False-1]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure[False-2]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure[False-3]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure[True-0]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure[True-1]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure[True-2]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure[True-3]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_unlisted_vmap_callback_absence_remains_absent_and_default_admission_unchanged
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[empty]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[missing-dot]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[duplicate]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[scenario]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[counterpart]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[digest]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[report]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[control-revision]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[counterpart-digest]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[raw-callback]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[check-status]
tests/extra_tests/converter/test_source_precision_corpus_replay.py::test_audited_control_invariant_cannot_pass_by_omission_or_corruption[extra-check]
tests/extra_tests/converter/test_source_finalization.py::test_valid_explicit_attribute_override_preserves_actual_jax_semantics[proto-graph]
tests/extra_tests/converter/test_source_finalization.py::test_valid_explicit_attribute_override_preserves_actual_jax_semantics[proto-nested]
tests/extra_tests/converter/test_source_finalization.py::test_valid_explicit_attribute_override_preserves_actual_jax_semantics[ir-graph]
tests/extra_tests/converter/test_source_finalization.py::test_valid_explicit_attribute_override_preserves_actual_jax_semantics[ir-nested]
tests/extra_tests/converter/test_source_finalization.py::test_valid_explicit_attribute_override_preserves_actual_jax_semantics[file-graph]
tests/extra_tests/converter/test_source_finalization.py::test_valid_explicit_attribute_override_preserves_actual_jax_semantics[file-nested]
tests/extra_tests/converter/test_source_finalization.py::test_nonconvertible_explicit_attribute_cannot_silently_keep_schema_default[proto-graph]
tests/extra_tests/converter/test_source_finalization.py::test_nonconvertible_explicit_attribute_cannot_silently_keep_schema_default[proto-nested]
tests/extra_tests/converter/test_source_finalization.py::test_nonconvertible_explicit_attribute_cannot_silently_keep_schema_default[ir-graph]
tests/extra_tests/converter/test_source_finalization.py::test_nonconvertible_explicit_attribute_cannot_silently_keep_schema_default[ir-nested]
tests/extra_tests/converter/test_source_finalization.py::test_nonconvertible_explicit_attribute_cannot_silently_keep_schema_default[file-graph]
tests/extra_tests/converter/test_source_finalization.py::test_nonconvertible_explicit_attribute_cannot_silently_keep_schema_default[file-nested]
tests/extra_tests/converter/test_source_profile.py::test_profile_distinguishes_inconsistent_traced_metadata[graph6-cast attribute]
tests/extra_tests/converter/test_source_profile.py::test_profile_accepts_integer_data_and_requires_boolean_cond_origin
tests/extra_tests/converter/test_source_precision_corpus_numerics.py::test_real_fixed_source_drift_blocks_before_runtime[dtype]
tests/extra_tests/converter/test_source_precision_corpus_numerics.py::test_real_fixed_source_drift_blocks_before_runtime[shape]
tests/extra_tests/converter/test_source_precision_corpus_numerics.py::test_real_fixed_source_drift_blocks_before_runtime[weak]
tests/extra_tests/converter/test_source_precision_corpus_numerics.py::test_real_fixed_source_drift_blocks_before_runtime[source]
tests/extra_tests/converter/test_source_precision_corpus_numerics.py::test_real_fixed_source_drift_blocks_before_runtime[axis]
tests/extra_tests/converter/test_source_precision_corpus_numerics.py::test_current_default_add_and_reviewed_ordinary_matrix_admission
tests/extra_tests/converter/test_source_native_cnn_nonzero.py::test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles[False-False-0]
tests/extra_tests/converter/test_source_native_cnn_nonzero.py::test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles[False-False-1]
tests/extra_tests/converter/test_source_native_cnn_nonzero.py::test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles[False-True-0]
tests/extra_tests/converter/test_source_native_cnn_nonzero.py::test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles[False-True-1]
tests/extra_tests/converter/test_source_native_cnn_nonzero.py::test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles[True-False-0]
tests/extra_tests/converter/test_source_native_cnn_nonzero.py::test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles[True-False-1]
tests/extra_tests/converter/test_source_native_cnn_nonzero.py::test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles[True-True-0]
tests/extra_tests/converter/test_source_native_cnn_nonzero.py::test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles[True-True-1]
tests/extra_tests/converter/test_source_native_cnn_pool.py::test_named_complete_original_cnn_keeps_same_symbolic_artifact[False]
tests/extra_tests/converter/test_source_native_cnn_pool.py::test_named_complete_original_cnn_keeps_same_symbolic_artifact[True]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_exact_five_raw_sources_prepare_one_instance_artifact_and_two_batch_feeds[False-case3]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_exact_five_raw_sources_prepare_one_instance_artifact_and_two_batch_feeds[False-case4]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_exact_five_raw_sources_prepare_one_instance_artifact_and_two_batch_feeds[True-case3]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_exact_five_raw_sources_prepare_one_instance_artifact_and_two_batch_feeds[True-case4]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_only_explicit_counterpart_types_can_supply_structure
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_real_v3_live_controls_reject_v1_and_stale_control_records
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_native_linear_counterpart_rejects_parameter_and_wiring_corruption[kernel]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_native_linear_counterpart_rejects_parameter_and_wiring_corruption[bias]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_native_linear_counterpart_rejects_parameter_and_wiring_corruption[port]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_native_linear_counterpart_rejects_parameter_and_wiring_corruption[cast]
tests/extra_tests/converter/test_source_leading_batch_corpus.py::test_native_linear_counterpart_rejects_parameter_and_wiring_corruption[missing]
tests/extra_tests/converter/test_source_fixed_parameter_models.py::test_exact_original_parameter_model_first_barrier[0]
tests/extra_tests/converter/test_source_fixed_parameter_models.py::test_exact_original_parameter_model_first_barrier[1]
tests/extra_tests/converter/test_source_fixed_parameter_models.py::test_exact_original_parameter_model_first_barrier[3]
tests/extra_tests/converter/test_source_fixed_parameter_models.py::test_exact_original_parameter_model_first_barrier[4]
tests/extra_tests/converter/test_source_fixed_parameter_models.py::test_exact_original_parameter_model_first_barrier[20]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_real_scan_function_conv_preserves_carry_and_sliced_input_shapes[False-carry]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_real_scan_function_conv_preserves_carry_and_sliced_input_shapes[False-xs]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_real_scan_function_conv_preserves_carry_and_sliced_input_shapes[True-carry]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_real_scan_function_conv_preserves_carry_and_sliced_input_shapes[True-xs]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_named_fp64_matrix_in_scan_keeps_actual_double_physical_shapes[False]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_named_fp64_matrix_in_scan_keeps_actual_double_physical_shapes[True]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_packed_complex_scan_carry_and_slices_retain_both_named_function_channels[False-complex64]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_packed_complex_scan_carry_and_slices_retain_both_named_function_channels[False-complex128]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_packed_complex_scan_carry_and_slices_retain_both_named_function_channels[True-complex64]
tests/extra_tests/converter/test_source_scan_physical_metadata.py::test_packed_complex_scan_carry_and_slices_retain_both_named_function_channels[True-complex128]
tests/extra_tests/converter/test_source_captured_precision.py::test_source_captures_specialize_weak_metadata_and_reuse_equal_states[weak-first-instance-False]
tests/extra_tests/converter/test_source_captured_precision.py::test_source_captures_specialize_weak_metadata_and_reuse_equal_states[weak-first-instance-True]
tests/extra_tests/converter/test_source_captured_precision.py::test_source_captures_specialize_weak_metadata_and_reuse_equal_states[strong-first-instance-False]
tests/extra_tests/converter/test_source_captured_precision.py::test_source_captures_specialize_weak_metadata_and_reuse_equal_states[strong-first-instance-True]
tests/extra_tests/converter/test_source_captured_precision.py::test_source_legacy_source_captures_leave_ordinary_jax_behavior_intact[instance]
tests/primitives/test_eqx.py::Test_weight_norm::test_eqx_weight_norm_linear_dynamic
tests/primitives/test_eqx.py::Test_weight_norm::test_eqx_weight_norm_linear
tests/primitives/test_eqx.py::Test_weight_norm::test_eqx_weight_norm_conv2d
tests/primitives/test_eqx.py::Test_conv::test_eqx_conv2d_nchw
tests/primitives/test_eqx.py::Test_conv::test_eqx_conv2d_batched_nchw
tests/primitives/test_eqx.py::Test_gru_cell::test_eqx_gru_cell_basic
tests/primitives/test_eqx.py::Test_lstm_cell::test_eqx_lstm_cell_basic
tests/primitives/test_eqx.py::Test_spectral_norm::test_eqx_spectral_norm_linear_inference
```

</details>

Evidence is retained locally under
`/tmp/jax2onnx-restart-checkpoint-20261021/`. These records are provenance rather
than tracked dependencies; no model or array payload is needed to read this
ledger. The receipt paths identify each result's exact source, selector and
scope. Phase counts are not inferred where drivers did not instrument them.

| Evidence | Retained record |
| --- | --- |
| Completed local closeout | `native-wrapper-checkpoint-closeout-v1/receipt.json`; `native-wrapper-documentation-qualification-v1/receipt.json`; `native-wrapper-committed-source-v1/source-inputs.json` |
| Prior green three-stack directed owner qualification and c5 typing | `source-final-constant-origin-actual-v1-primary/receipt.json`; `source-final-constant-origin-actual-v1-secondary/receipt.json`; `source-final-constant-origin-actual-v1-minimum/receipt.json`; `source-c5-current-typing-v1/receipt.json` |
| Current code cut and permanent repeated-Function/If owner | `source-integer-cse-helper-typing-root-application-v1/source-inputs.json`; b90 live-branch packet `source-integer-cse-live-branch-root-application-v1/source-inputs.json`; prior unit-FMM repair `source-unit-fused-matmul-refresh-root-application-v1/source-inputs.json`; prior bootstrap `source-initial-inference-provenance-root-application-v1/source-inputs.json`; prior fresh-symbol packet `source-fresh-inference-symbol-root-application-v1/source-inputs.json`; prior fixture packaging `source-post-restoration-type-refresh-fixture-root-application-v1/source-inputs.json`; generic refresh `source-post-restoration-type-refresh-root-application-v1/source-inputs.json`; earlier marker-only packaging `source-repeated-function-if-main-root-application-v1/source-inputs.json`; `source-repeated-function-if-marker-packaging-v1/manifest.json`; `astra-repeated-function-if-main-application-amendment-v1/amendment.json` |
| Applied generic refresh, exact review and current actual gate | `source-post-restoration-type-refresh-sealed-v1/manifest.json`; `astra-post-restoration-type-refresh-independent-review-v1/review.json`; `source-post-restoration-type-refresh-root-application-v1/receipt.json`; `source-post-restoration-type-refresh-actual-v1-primary-primary/receipt.json` |
| Applied exact typing-only correction and fresh 9134 typing/RNG | `source-wip-fixed-typing-v1/receipt.json`; `source-integer-cse-helper-typing-sealed-v1/manifest.json`; `astra-integer-cse-helper-typing-review-v1/review.json`; `source-integer-cse-helper-typing-root-application-v1/receipt.json` |
| Actual failed b90 typing attempt; RNG not run | `source-b90-wip-current-typing-v1/receipt.json` |
| Prior e6 and earlier source51/a8/f82 typing/RNG | `source-e6-current-typing-v1/receipt.json`; `source-51-current-typing-v1/receipt.json`; `source-a8-current-typing-v1/receipt.json`; `source-f82-current-typing-v1/receipt.json` |
| Retained f82 six-owner three-stack 183 gate | `source-refresh-fixture-current-actual-v1-primary/receipt.json`; `source-refresh-fixture-current-actual-v1-secondary/receipt.json`; `source-refresh-fixture-current-actual-v1-minimum/receipt.json` |
| Applied exact fixture packaging and separate retained-symbol observation | `source-post-restoration-type-refresh-fixture-packaging-v1/manifest.json`; `source-post-restoration-type-refresh-fixture-root-application-v1/receipt.json`; `source-refresh-symbol-provenance-retained-v1/safe-summary.json`; `source-refresh-symbol-provenance-retained-root-launch-v1/receipt.json` |
| Applied fresh-symbol packet, exact review and current actual gates | `astra-fresh-inference-symbol-contract-v1/contract.json`; `source-fresh-inference-symbol-sealed-v1/manifest.json`; `astra-fresh-inference-symbol-review-v1/review.json`; `source-fresh-inference-symbol-root-application-v1/receipt.json`; `source-fresh-inference-symbol-actual-v1-primary/receipt.json`; `source-fresh-symbol-current-actual-v1-minimum/receipt.json`; `source-fresh-symbol-current-actual-v1-secondary/receipt.json` |
| Applied initial-symbol bootstrap contract | `astra-initial-inference-provenance-contract-v1/contract.json` |
| Prior source51 bootstrap application and actual gate | `source-initial-inference-provenance-sealed-v1/manifest.json`; `astra-initial-inference-provenance-review-v1/review.json`; `source-initial-inference-provenance-root-application-v1/receipt.json`; `source-initial-inference-provenance-actual-v1-primary/receipt.json` |
| Prior source51 paired 223 directed selections | `source-initial-provenance-current-actual-v1-secondary/receipt.json`; `source-initial-provenance-current-actual-v1-minimum/receipt.json` |
| Applied unit-FMM selection and actual e6 gates | `astra-unit-fused-matmul-refresh-contract-v1/contract.json`; `source-unit-fused-matmul-refresh-sealed-v1/manifest.json`; `astra-unit-fused-matmul-refresh-review-v1/review.json`; `source-unit-fused-matmul-refresh-root-application-v1/receipt.json`; `source-unit-fused-matmul-refresh-actual-v1-primary/receipt.json`; `source-unit-fmm-current-actual-v1-secondary/receipt.json`; `source-unit-fmm-current-actual-v1-minimum/receipt.json` |
| Applied exact live-branch proof and current b90 joint gate | `source-integer-cse-live-branch-sealed-v1/manifest.json`; `astra-integer-cse-live-branch-review-v1/review.json`; `source-integer-cse-live-branch-root-application-v1/receipt.json`; `source-integer-cse-live-branch-actual-v1-primary/receipt.json` |
| LOCAL WIP gate authority, not accepted-checkpoint evidence | `root-local-wip-checkpoint-gate-planning-v1.json` |
| Actual failed ordinary b90 full suite and corrected54+2 summary | `source-integer-cse-current-full-v1-primary/receipt.json`; `source-integer-cse-current-full-v1-primary-xml-metadata-summary-v2/summary.json`; `root-terminal-b90-failure-identities-safe-v2/summary.json`; preserved v1 `root-terminal-b90-failure-identities-safe-v1/summary.json` |
| Fresh current 9134 six-owner SECONDARY/DIRECTMIN gates | `source-wip-ir10-owners-v1-secondary/receipt.json`; `source-wip-direct-minimum-owners-v1-minimum/receipt.json` |
| Read-only after-checkpoint full-scope audit, not executed qualification | `astra-after-checkpoint-scope-audit-v1/audit.md` |
| Retained e6 final-contract delta and diagnostic consumer groups | `source-final-boundary-contract-delta-retained-v1/safe-summary.json`; `source-final-cast-consumer-groups-retained-v1/safe-summary.json` |
| Retained source51 Gemm failure and skipped unit-FMM producer chain | `source-bootstrap-gemm-failure-retained-v1/safe-summary.json`; `source-bootstrap-gemm-producer-chain-retained-v1/safe-summary.json` |
| Retained a8 failure and corrected inference-call attribution | `source-fresh-symbol-later-failure-retained-v1/safe-summary.json`; `source-fresh-symbol-transition-retained-v1/safe-summary.json`; earlier `source-refresh-symbol-provenance-retained-v1/safe-summary.json` |
| Earlier Transformer metadata boundary and unchanged refresh contract | `source-slice-shape-combined-actual-v1-primary/receipt.json`; `source-slice-shape-later-boundary-retained-metadata-v1/safe-summary.json`; `astra-unit-fmm-post-restoration-type-refresh-contract-v1/contract.json` |
| Earlier state, transport and target qualification | `source-shared-models-b46-floor-minimum/receipt.json`; `source-deployment-correspondence-and-vit-v1-primary/receipt.json`; `source-native-nnx-log-softmax-root-application-v1/source-inputs.json` |
| Integer cross-stack outcomes and safe historical seven errors | `source-integer-v1-cross-stack-qualification-ledger/receipt.json`; `source-integer-minimum-seven-safe-exception-extract-v1/receipt.json` (SHA256 `c324423c18ab49e6a7b6a9572856b4376c38b47abc54ad73c513e20c8380b1cf`) |
| Cast571 and new original7-on acceptance | `source-positive-symbolic-cast-test-corrections-v2-focused-primary/receipt.json`; `source-positive-symbolic-cast-v1-original7-x64-on-primary/receipt.json`; durable evidence manifest `source-positive-symbolic-cast-v1-original7-x64-on-evidence-primary/evidence-manifest.json` |
| Recovered original7/8/9 terminal observation | `source-original-nested-function-owners-v1-terminal-observation/receipt.json` (SHA256 `4a2802c9cb5f432b2107ac87a5da5c13d8b19c8786cd2d667fe8d54ecbb0df64`); all six cells retained in `source-original-nested-function-owners-v1-evidence-primary/` |
| TopK PRIMARY and corrected SECONDARY selection | `source-topk-min-max26-v1-focused-primary/receipt.json`; `source-topk-minmax-corrected-v1-focused-secondary/receipt.json` |
| PRIMARY Min/Max pass and activation failure | `source-activation-min-max26-corrected-v1-focused-primary/receipt.json`; safe diagnosis `source-activation-zero-sign-runtime-diagnostic-v1/receipt.json` (SHA256 `c5e5341861c3582c2dd181567fce02caa0ff32ece80a6b8e71a7271f6b8ba359`) |
| Eqx18 complete original pair and controls | `source-eqx18-default-anchor-v1-full-original-primary/receipt.json`; `source-eqx18-default-anchor-v1-full-original-secondary/receipt.json`; durable manifests under `source-eqx18-default-anchor-v1-full-original-evidence-{primary,secondary}/` |
| Linen19/Spectral20 and earlier Eqx default failures | `source-original-dropout-spectral-owners-v2-primary/receipt.json`; retained initial zero-case selector attempt remains separate |
| Original GPT both-stack results and safe status | `source-native-unused-input-v1-original-gpt-primary/receipt.json`; `source-native-unused-input-v1-original-gpt-evidence-primary/safe-status.json` (SHA256 `6c0fd94452d6111573f9853eec53f0522b052f4427e6bd5f3898d05e8380f74d`); `source-native-unused-input-v1-original-gpt-secondary/receipt.json`; SECONDARY safe-status SHA256 `211e70ac13e3e471b281e70bcc845b0395234e675b1b56ccd8fc5b385a9ab676` |
| Full original Transformer29 retained boundary | `source-min-max26-topk-v1-full-original-transformer-primary/receipt.json`; `source-min-max26-topk-v1-full-original-transformer-evidence-primary/safe-status.json` (SHA256 `1d7464da5e0e0d45bcaf5c0d50abc4e0fe0648872885c25491ea5574e4e5d7d2`) |
| Earlier typing/RNG | `source-zero-topk-fusion-typing-correction-v2/receipt.json`; `source-84e-hotfix-ir11-current-typing-v1/receipt.json`; `source-c6-hotfix-ir11-current-typing-v1/receipt.json`; earlier `source-dcf-current-typing-v1/receipt.json` |
| Applied capability packet and independent review | `source-installed-capability-diagnostics-draft-v1/manifest.json`; `astra-installed-capability-diagnostics-v1-review/review.json` (SHA256 `0552d4a92d0ce5d509d4bec81b627d6688240b762fa5c00fd209331cbbefbadf`) |
| Applied Transformer scanner correction | `source-transformer-softmax-artifact-provenance-correction-v1/manifest.json`; `astra-transformer-softmax-artifact-provenance-v1-review/review.json` (SHA256 `51b8bec9e505bd3a5f0ee7e77496b8d76ec8b8a05782be6cfa5b0f88a79aa1d6`) |
| New environment and explicit native-probe epoch | `primary-hotfix-environment-observation-v1/source-inputs.json`; `source-topk-primary-hotfix-native-protocol-v1/receipt.json` (SHA256 `4d6bb13a85a686be1765be2ce752fa14f625c60fd950aeaad8453567253b4393`); `source-topk-remaining-widths-native-secondary-v1/safe-summary.json`; `source-topk-remaining-widths-native-probe-sealed-v2/manifest.json` |
| Updated PRIMARY core/capability controls | `source-onnx-ir11-core-capability-v1-focused-primary/receipt.json` (SHA256 `2a8fb943cbea97bdf4a01da77c89387ae6528f326946a23920e05945100f3e60`) |
| Dependency strict documentation gates | `source-c99-dependencies-docs-build-root-launch-v1/receipt.json`; earlier `source-user-poetry-dependencies-doc-v1-strict-docs/receipt.json` (SHA256 `594bdc879e279698be541324d6855311b18d28e1629c9dfaca776d0e01a6b1d2`) |
| Dependency documentation and current source cut | `source-user-poetry-update-dependencies-doc-v1/manifest.json`; `source-integer-cse-helper-typing-root-application-v1/source-inputs.json` |
| Matched DIRECTMIN core/original selection and later precise closure | `source-onnx-ir10-core-capability-v1-focused-minimum/receipt.json`; later `source-missing-schema-counterpart-v3-focused-primary/receipt.json` and `source-missing-schema-counterpart-v3-focused-minimum/receipt.json` |
| SECONDARY diagnostics/tensor selection and IR return modes | `source-ir10-secondary-capability-tensor-v1-secondary/receipt.json`; `source-ir11-return-modes-v1-primary/receipt.json`; `source-ir10-return-modes-v1-minimum/receipt.json` |
| Actual BF16 retained-category and post-observation explanations | `source-topk-bf16-retained-word-analysis-v1/safe-summary.json`; `source-topk-bf16-post-observation-canonicalization-diagnostic-v1/safe-summary.json` |
| Retained paired BF16/alias/deployment gates | `source-bf16-alias-deployment-actual-v1-primary/receipt.json`; `source-bf16-alias-deployment-actual-v1-secondary/receipt.json`; `source-bf16-alias-deployment-actual-v1-minimum/receipt.json`; approved BF16 packet `source-topk-portable-bfloat16-continuation-v1/manifest.json`; review `astra-topk-portable-bfloat16-independent-review-v1/review.json` |
| Retained paired BF16 strict native-equivalence outcomes | `source-bf16-primary-actual-equivalence-rollup-v2.json` (SHA256 `cb3e60d62c412e2f420ce853bd2a97365a13862f42884a448e2cf7506f382014`); `source-bf16-secondary-actual-equivalence-rollup-v2.json` (SHA256 `5c700fd59d3e661b7ecb28e0d4a5ddede5cb484f3c1ee0b13932d124d5ed43bd`); nine/eight actual comparison record hashes, no array-payload reads |
| Historical repeated Function/If export ownership failure | `source-repeated-function-if-actual-v1-primary/receipt.json`; later c5 unchanged external2 passes all three stacks, and dd1 packages the identical semantic body in MAIN |
| Historical typing failure and narrow correction | `source-78ed-constant-alias-bf16-current-typing-v1/receipt.json`; `source-topk-redundant-cast-typing-correction-v1/manifest.json`; `source-a893-topk-typing-correction-current-typing-v1/receipt.json` |
| Width, empty and Function-alias source boundaries | `source-topk-global-dispatch-actual-gate-v1-primary/receipt.json`; `source-topk-empty-result-actual-gate-v2-primary/receipt.json`; `source-constant-alias-actual-regressions-v1-primary/receipt.json`; `astra-function-optimized-output-alias-independent-review-v1/review.json` |
| Actual trace-only supplement and admitted preparation review | `source-topk-native-trace-primary-v1/safe-summary.json`; `source-topk-native-trace-secondary-v1/safe-summary.json`; `source-topk-trace-root-preparation-review-v1.json` |
| Actual shape-only HLO and retained extraction gaps | `source-bf16-topk-shape-only-hlo-primary-v1/safe-summary.json`; `source-bf16-topk-shape-only-hlo-secondary-v1/safe-summary.json`; `astra-bf16-hlo12-actual-safe-review-v1/review.json`; `source-bf16-topk-root-retained-gap-classification-v1.json` |
| Immutable earlier-source map and finite retained review | `product-map-c686-eightcells-v2/receipt.json`; `astra-eqx-actual-numerical-review-v3/review.json`; `source-original14-first-whole-acceptance-evidence-v1/evidence-manifest.json` |

Raw metadata and separately frozen numerical evidence remain distinct. Original
MLP0/2 retain explicit `skip_numeric_validation=True`; MLP1 retains its omitted,
default-false flag. All three raw rows retain `criteria=None`, failed structural
callbacks and artifact-only runtime/numerics **UNVERIFIED**. Separate actual
finite acceptance does not overwrite those fields. The original GPT criterion
**67cd821a** and prepared Transformer criterion **d2ecaed3** are unchanged. The five native-binding owner indices 0/1/3/4/20 now have artifact-checked
full-model evidence in their frozen gate; standalone raw `criteria=None` and
four UNVERIFIED fields remain unchanged. No blanket numerical acceptance or
failure waiver follows. Known
ordering, subnormal, FP16-rounding and optimized-runtime failures remain
**FAILED**; missing evidence remains **UNVERIFIED**.

The same ordinary full suite is terminal and failed; its exact totals and 116
identities are now recorded above. All 56 retained selector dispositions pass: **54 model/state runs plus 2 guards**.
The historical 56-selector result of **19 PASS / 37 unfinished** is not current
qualification. The typing-only correction is applied at 9134; fresh typing/RNG and both scoped
SECONDARY/DIRECTMIN selections pass. Complete strict docs/rendered links, all
seven normal hooks and exact staged review for
the explicitly unfinished LOCAL WIP commit. Unresolved real failures remain
recorded and do not themselves prohibit preserving reviewed WIP. Keep the
fixture/expectation candidates external/unapplied; do not start another broad
full run just to make WIP preservation contingent on repairing all 116 failures.
Record actual preservation-commit check, strict-docs, normal-hook and staged
review outcomes in the normal commit body; none converts the failed full suite
to an accepted milestone.

Even a green full standard suite would close only this checkpoint gate. The
read-only after-checkpoint scope audit retains four further packets: execute
the complete current-source 3,704-cell map and optional discovery; close measured
implementation and separate 63 opaque/104 metadata-free/752 original manual
accounting; finish all fixed-source preparation and numerical/installed
evidence; and implement the coordinated ordinary export, allclose, browser and
generator/public-preparation cutover with final acceptance. These are execution
and implementation obligations, not completion implied by 331 PASS, a LOCAL WIP
commit, an ordinary
full suite or 3,704 product terminals. No new controller or duplicate 56 gate is
needed; reuse the existing tools and each result’s exact scope.

Preserve paired directed results and strict BF16 native-equivalence failures
as separate evidence. Complete the supported-stack/minimum qualification,
current inventory and complete original models under their frozen criteria.
The full **1,852-row / 3,704-caller** current-source map, all **22 original owners**,
**50 optional rows**, **104 registrations without metadata**, all fixed supplied
JAX sources, scientific FP64/low-precision evidence and phases B–F remain
required. The immutable c686 map’s **106 ordinary terminals, one incomplete and
3,597 unstarted** belongs to its earlier source; those outcomes do not qualify
the current map or numerical behavior. The held four-file namespace with 20 scoped preparation exports and its installed
smoke remain obligations; it is not installed/published by this WIP. Only
demonstrated ONNX representation and actual target limits may define support boundaries; unfinished conversion
work cannot become a permanent exclusion.

Public precision behavior remains **LEGACY** until the complete uniform source-preserving
contract is qualified across all existing functionality and fixed supplied JAX sources. No accepted full-current qualification, denominator reduction,
permanent public selector, per-operation fallback or capability loss is implied.

## Retained checkpoint evidence

The sections below preserve their original source epochs, failures and criteria.
Their then-current language is historical; the current ledger above controls
status. Numerical samples never transfer automatically to changed code.

The authorized support floor is **Python >=3.11 / JAX and JAXLIB >=0.10.2 /
Flax >=0.12.8**. Dependency declarations and both checked lockfiles are applied;
retained metadata from an unpublished pre-restart wheel declares those bounds.
Existing locked package versions and Python variants are retained. Python 3.11 remains supported. The
SECONDARY stack is the exact framework-floor target; its newer ONNX/ORT and
other dependencies do not establish all-direct-dependency-minimum acceptance.
ORT 1.24.1 and the other declared bounds are unchanged.

Earlier JAX 0.8.1/Flax 0.12.1 runs retain their actual outcomes as historical
evidence; they are no longer a supported-floor release gate. This decision
does not fix numerical/provider gaps or remove any existing model, fixed source
or numerical requirement from the single-interface migration target.


### Native transformed-callable capture

The fe214590 checkpoint replay boundary captures the original native `vmap_f` and
`reraise_with_filtered_traceback` functions by exact nested-code and defining-
globals identities. It retains the actual wrapper/source closures, owners and
aliases without invoking the source or rebuilding a transformation during
capture. Finite implementation dependencies, class/default metadata, actual
function builtin mappings and true builtin fallback absence/resolution remain
owned; this does not grant general callable or carrier admission.

Both supported JAX layouts resolve `vmap_f`'s direct `zip` through the imported
native `safe_zip`. Capture requires the same exact object at `api.zip`,
`api.util.safe_zip` and `api.util.jaxlib_utils.safe_zip`, retaining and rechecking
all defining module edges. Removing the global or substituting builtin `zip`
is a state change; it cannot silently become a true builtin fallback.

Native vmap can materialize an unset axis local as the already effective axis
environment. Selected replay operations therefore install that same effective
axis as a managed local, require the managed runtime state to remain unchanged,
then restore the exact original caller local and effective state in `finally`.
The original native setter is retained across the call. Native returns/errors,
exported artifacts/schema outcomes and raw callback statuses are recorded before
cleanup; unexpected context drift remains a separate unverified replay gap even
if caller restoration succeeds. Source weights, globals and closures are not
reset or excluded from state checks.

**Historical focused runtime evidence before the typing-only correction:**
**441 PRIMARY / 437 SECONDARY / 437 MINIMUM cases PASS**, zero failures/errors/
skips, across the wrapper, canonical native capture and raw-replay owners.
These tests include positive, rejection and diagnostic outcomes; phase counts
are NOT_INSTRUMENTED. Failed earlier builtin-layout, descriptor-restoration,
axis-local and interned no-op fixture epochs remain retained. The subsequent
annotations/casts-only compat correction preserves normalized runtime AST;
actual production mypy passes all **559 files**, followed by the actual inventory
of **17 RNG factories**. Inventory is not RNG or model numerical acceptance.
The same typed sources pass both 15-owner affected selections: **919 SECONDARY
and 919 DIRECTMIN cases**, zero failures/errors/skips, with source/Git state held.
The committed fe214590 PRIMARY suite passes **12,762 unique cases**, zero failures/
errors/skips, in **1458.641726 seconds**, with source/Git state held. Exact identity
reconciliation retains all 12,600 preceding full-suite cases and adds only 162
wrapper-owner cases; neither prior IDs nor affected-selection IDs are removed.
SECONDARY and DIRECTMIN have the same 919 identities, all present in full PRIMARY.
All 1,109 source pins hold and the owned native process group is empty. Phases
remain **NOT_INSTRUMENTED**. Strict docs/rendered links, all seven actual normal
commit hooks and qualified/staged-byte verification subsequently completed in
local commit `fe214590`. These passing regression results do not establish full
B–F or model numerical acceptance.

**Original MLP2 result.** Under both original caller x64 states on PRIMARY,
SECONDARY and MINIMUM, the unchanged ordinal-2 `MlpExample` target is captured;
eager/fresh-JIT and both typed traces return, and source reference, export,
schema, artifact and lowering-audit checks validate. Its original structural
callbacks remain **FAILED**, `criteria=None` and `skip_numeric_validation=True`
are unchanged, and runtime/numerics remain **UNVERIFIED**. Raw replay runs no
ORT. Passing regression assertions preserve that failed callback outcome.

Original metadata differs per row: MLP0 and MLP2 explicitly set
`skip_numeric_validation=True`; MLP1 omits that flag and retains its default
false. All three raw rows keep `criteria=None`; artifact-only replay executes
no ORT and leaves runtime/numerics **UNVERIFIED**. Those raw fields do not replace
separately reviewed numerical criteria or certificates.

**Separate source-only context evidence.** The accepted v4 capture covers four
original unbatched MLP0/1 training/inference caller cells at the earlier
`b945a957` source snapshot. Each cell has one traced source call and no concrete,
compiled or ORT outputs. That acceptance binds original files, state and context
to that exact epoch; later wrapper changes require refreshed correspondence.
It does not authorize a current numerical harness or clear the separate R2
numerical-execution hold. Original callbacks/criteria remain unchanged.

The independent immutable c686 map now retains **98 ordinary terminal cells /
one original incomplete / 3,605 unstarted of 3,704**, covering all 1,852 rows.
This remains earlier-source product coverage, not current runtime/numerical or
full-map acceptance. Public precision policy/options/defaults remain **LEGACY**;
all existing functionality, fixed sources and full B–F remain required through
one ordinary export contract.

Local evidence below is provenance, not a tracked dependency:

| Scope | Actual record |
| --- | --- |
| Historical pretyping focused runtime | `native-wrapper-axis-context-v3-three-stack/receipt.json` |
| Current typing-only source and actual typing/RNG | `native-wrapper-typing-application-v1/receipt.json` and `source-inputs.json`; `native-wrapper-typing-qualified-v1/receipt.json` |
| Current typed-source affected selections and full PRIMARY | `native-wrapper-final-v1-secondary/receipt.json`; `native-wrapper-final-v1-minimum/receipt.json`; `native-wrapper-final-v1-primary/receipt.json`; aggregate `native-wrapper-final-v1-controller/receipt.json` and `identity-reconciliation.json` |
| Original MLP2 source/export and retained failed callbacks | `native-wrapper-original-mlp2-observations-v1/summary.json` and its six original scenario records |
| Earlier four-cell source-only context, no numerical execution | `source-context-v4-root-execution-v1/receipt.json`; `astra-eqx-actual-source-context-review-v4/review.json` |
| Earlier-source map continuation | `product-map-c686-eightcells-v1/receipt.json` |

### Native Equinox Dropout and shared static uniform

#### Storage-v3 local closeout

After the qualification cut retained below, storage-v3 completed strict MkDocs,
rendered-link checks and all seven actual normal commit hooks. The exact qualified
paths were committed locally as `dde196609d4eb96f5827343406af160a2d3393da`, preserving
all 13 Python postimages, 1,108 source/config inputs and the four protected docs.
The closeout is local only, with no push. These completed gates belong to that
storage-v3 checkpoint; current wrapper sources require their own gates above.

#### Storage-v3 qualification snapshot

This preceding source snapshot retains its own outcomes and then-pending gates.
The later wrapper and source-only context results above supersede its future-
wrapper wording; none of its results automatically transfer to changed sources.

The storage-v3 qualification snapshot has **13 Python paths / 1,108 source/config inputs**,
bound by `source-function-storage-application-v3/source-inputs.json`. Public
precision policy/options/defaults stay **LEGACY**. The shared serializer also
corrects ordinary public `return_mode="file"` standard exports, retaining root
external storage and inline Function tensor attributes across ONNX versions.
Exact tensor payloads, source/options/state/aliases and original criteria remain
binding. At the qualification cut recorded here (2026-10-21T20:59:01.968117+00:00), strict
documentation, rendered-link checks, normal commit hooks, staging/qualified-byte
checks and commit had not yet run. Later closeout evidence is recorded separately.
C686's seven actual hooks were completed for that earlier committed checkpoint.

**Current source-bound results.** The same five storage cases pass on direct
minima and PRIMARY in 8.879526 and 10.193245 seconds, zero failures/errors/skips.
All seven direct quality checks pass on the 13 Python paths and 1,108 inputs,
including 559-file typing/RNG inventory, pinned-hook Black, Ruff and four
policies. These are direct checks, not normal commit hooks. Phases in the
native receipts are **NOT_INSTRUMENTED**. Passing owner tests also include
genuine rejection/diagnostic assertions; failed raw records are not relabeled.

| Current source-bound execution | Actual result | Duration |
| --- | --- | ---: |
| Direct-minimum supported partition | 1,752 PASS; zero failures/errors/skips | 268.164035 seconds |
| SECONDARY, all 26 owners | 1,813 PASS; zero failures/errors/skips | 286.059170 seconds |
| PRIMARY, all 26 owners | 1,813 PASS; zero failures/errors/skips | 242.660132 seconds |
| Full PRIMARY | 12,600 PASS; zero failures/errors/skips | 1451.264381 seconds |

The current population is **1,813 = 1,752 supported minimum cases + 61 exact
target-capability cases**. The 61 comprise 59 actual newly observed target
rejections (58 ORT opset-26 cases above the minimum runtime's supported ceiling
and one missing BitCast schema) plus two prior cases. Their exact identities/
reasons remain in the reviewed capability partition; the original 181-case
ledger is unchanged. They are not passes, waived requirements or removed
functionality. All 1,813 identities run on each modern stack. The full PRIMARY suite records **12,600 PASS / 0 FAIL / 0 ERROR / 0 SKIP of 12,600 cases**, in 1451.264381 seconds.
The seven-stage ROOT serial qualification is terminal PASS with all 1,108
source inputs held. Exact XML identity reconciliation retains all 12,416 prior
full-suite identities, removes none and adds 184 (16 Dropout-parameter, 100
Equinox Dropout, one facade, 51 literal-cond and 16 raw-replay cases). All
1,813 modern affected-owner identities occur in the full suite. These test
results do not supply instrumented phase counts or full B–F acceptance.

**Retained failures and storage scope.** The preceding expanded minimum run
remains **1,750 PASS / 60 FAIL of 1,810**, zero errors/skips: 59 target-capability
failures and one real Function storage-contract failure, independently reproduced
as one FAIL at unchanged c686. Both original 1.2 MB FP64 payloads, including
signed zeros/NaN bits, were already exact, and full loaded-model checker passed.
The failure was required Function-inline storage, not lost tensor bits. The
public ONNX toy retains this original minimum contract failure; the candidate
matches the modern root-external/Function-inline storage with exact payloads.
That toy uses no runtime sessions and has UNVERIFIED numerics.

Storage fixture v1 and v2 each retain **four PASS / one FAIL of the same five**.
Only the new LEGACY fixture failed before save: unsupported tuple-return Function,
then input/output alias-name conflict. Test-only single-output module wrappers
and consistent alias naming retain all payload/storage assertions. Production
UI stays `bd79b657…` through those fixture revisions; v3 facade tests are
`da71e691…`. Current five-case success does not rewrite those failures.

The literal-helper v2 25-PASS/29-FAIL and v3 45-PASS/nine-FAIL runs remain failed;
the later initial 54-case v4 pass and seven quality passes retain their own epoch.
The independently accepted 64-call artifact diagnostic retains eight full
checkers, 512 saved outputs and 384 before/after owners: original 32 calls have
16 mismatched output occurrences; duplicate-Identity variants match all 256
outputs across 32 calls under both settings, without owner mutation. It informed
the SOURCE result-boundary repair and is distinct from production qualification.
The lifetime repair's same six-case three-stack passes, earlier 680-case modern
passes / minimum 675-PASS/five-FAIL and preliminary quality failures also remain.
No source, flag, runtime setting, callback, criterion or baseline is rewritten.

All four original MLP0/1 callbacks remain **FAILED**, criteria None and runtime/
numerics **UNVERIFIED**; raw replay runs no ORT. MLP2's actual pre-native class
barrier remains its recorded outcome until a new source-held replay establishes
otherwise. Complete-model/low-precision/scientific-FP64 and fixed-source B–F
acceptance remain open through one ordinary export contract, without permanent
SOURCE/profile selectors, per-operation fallback or capability loss.

The independent immutable `c686304c` map now has **87 ordinary terminal cells,
one original index-70 incomplete attempt and 3,616 unstarted of 3,704**, retaining
all 1,852 rows, attempts and frozen source pins. These product observations do
not qualify current MAIN runtime/numerics, which remain UNVERIFIED.

| Separate future work | Exact current limit |
| --- | --- |
| Wrapper v4 / callable preparation v4 | Conditional static wrapper acceptance only; three-owner packets remain unapplied/natively unrun. Counts 136/140 and 74 are AST estimates, not collection or success. |
| Numeric certificates v2 | Source-only construction accepted, with 31 stdlib tests/80 independent endpoints; hard R2 numerical-execution hold remains. |
| Context PF1/PF2 v2 | Both findings resolved and source-only implementation conditionally accepted after 44 stdlib/16 independent positive/7 prior-counterexample controls. No actual capture or execution approval; fresh completed source/HEAD/pins and independent delta review are required. R2 remains. |

These packets do not delay current qualification. All existing functionality,
fixed sources, original callbacks/criteria, optional rows and metadata-free
anchors remain required. No static/source-only review establishes numerical
acceptance or a public precision cutover.

Evidence is local provenance under `/tmp/jax2onnx-restart-checkpoint-20261021/`;
paths are not tracked dependencies.

| Evidence | Receipt or independent record |
| --- | --- |
| Later storage-v3 local closeout at `dde196609d` | `source-storage-checkpoint-closeout-v1/receipt.json`; exact staged paths in `qualified-staged-paths.json`; seven actual hooks in `normal-commit.log` |
| Current source epoch and exact storage fixture revisions | `source-function-storage-application-v3/receipt.json` and `source-inputs.json`; `source-function-storage-focused-v{1,2,3}-minimum/receipt.json`; `source-function-storage-focused-v3-primary/receipt.json` |
| Current direct quality / wider runs | `source-function-storage-quality-v3/receipt.json`; `source-function-storage-affected-v3-{minimum,secondary,primary}/receipt.json`; full PRIMARY: `source-function-storage-full-v3-primary/receipt.json`; `source-function-storage-serial-qualification-v3/receipt.json`; `source-function-storage-full-identity-reconciliation-v1.json` |
| Failed expanded minimum and unchanged-c686 storage | `source-literal-cond-affected-v4-minimum/receipt.json`; `source-large-capture-c686-baseline-v1-minimum/receipt.json`; `astra-source-literal-cond-affected-v4-failure-audit-v1/audit.json` and `capability-partition.json` |
| Public storage diagnostic, numerical/runtime phases unrun | `source-function-storage-toy-root-v1/receipt.json` |
| Literal 54-case / artifact 64-call / lifetime six-case source epochs | `source-literal-cond-initial-v4-minimum/receipt.json`; `astra-minimum-cond-alias-probe-actual-v1/audit.json`; `mlp-runtime-feed-owner-retention-three-stack-v1.json` |
| Immutable product map | `product-map-c686-threecells-v5/receipt.json` |
| Future wrapper / certificate / context boundaries | `astra-native-wrapper-capture-v4-review-v1/review.json`; `astra-eqx-numeric-implementation-review-v2/review.json`; `astra-eqx-source-context-preflight-review-v2/review.json` |

#### Retained earlier Dropout and diagnostic epochs

The following text preserves the prior source epochs, their observations and
their then-current pending gates. The current cut and separate future packets
are stated above; historical pending statements do not override that cut.

The current private addition is the exact six-path Dropout v3 packet on
`c686304c6d9395be9e3df3dd3770c5a99ea1c2f5`. The packet manifest is
`eqx-dropout-forwarding-draft-v3/manifest.json`; its patch SHA256 is
`4dc39896cfcd29eed9339530b3d1c5219dd2c256ad3bb768c51362bcb1a884c6`.
Paths and receipts in this section are under the external evidence root
`/tmp/jax2onnx-restart-checkpoint-20261021/`. Public behavior stays **LEGACY**.

**Production scope.** Under private SOURCE policy, Equinox Dropout calls its
captured original `__call__` with the original positional/keyword arguments
before any legacy key or flag normalization. Original results, errors/warnings,
owner/key/input aliases and parameter storage remain observable. Its LEGACY
body and the existing private carrier/lowerer retain their behavior. This does
not replace the native source with an ONNX Dropout operation.

Shared uniform admission replaces the artificial static `(32,)` singleton
with general nonnegative static rank-one length. There is no special case for
the original MLP width 20. The complete native 13-equation uniform and
2-equation Bernoulli proofs, selected Threefry implementation and placement,
key/value/configuration checks, FP32/FP64 restrictions, requested opset/schema
checks and signed-INT64 count bound remain required. Existing rank-two/rank-three
admission remains; symbolic rank-one and rank-four negatives remain rejected.
The 31/33 identities are retained and now test genuine positives.

**Finite native and target evidence.** The new owner preserves eager, warmed
JIT and fresh JIT comparisons, original/activated typed traces and constants,
exact call forms, missing/malformed-key errors, aliases, opposite cache-policy
orders, native vmap/cond/loop capture and private-carrier/implementation/context/
placement/opset/count negatives. Static lengths **0, 1, 3, 20, 31, 32 and 33**
use the same admission rule. The independent scalar Python Threefry2x32-20
oracle is calibrated on its zero vector. Full uniform checks compare every
FP32/FP64 mantissa bit with exact dyadic expectations using one random consumer
per key/run; this is separate evidence from the Dropout mask/scaling criterion.

In the earlier exact v3 epoch, all three actual runs execute the same
**100 unique new-owner identities**
with **zero failures, errors or skips**, preserving all **1,107 source/configuration
pins** within each run. Phases are **NOT_INSTRUMENTED**.

| Actual execution | Framework / target | Result | Duration |
| --- | --- | --- | ---: |
| PRIMARY `eqx-dropout-v1-primary/receipt.json` | Python 3.12.9; JAX/JAXLIB 0.12.0, Flax 0.12.10; ONNX 1.23.2 / ORT 1.31.0 | 100 PASS | 78.439470516 seconds |
| SECONDARY `eqx-dropout-v1-secondary/receipt.json` | Python 3.11.12; JAX/JAXLIB 0.10.2, Flax 0.12.8; ONNX 1.23.2 / ORT 1.31.0 | 100 PASS | 102.724135493 seconds |
| Actual direct minima `eqx-dropout-v1-minimum/receipt.json` | Python 3.11.12; JAX/JAXLIB 0.10.2, Flax 0.12.8, Equinox 0.13.1; ONNX 1.20.1 / ORT 1.24.1 | 100 PASS | 84.395578746 seconds |

This finite owner result is not a full-stack or all-direct-minimum
existing-functionality claim. Actual minimum runtime capability remains distinct
from checker/schema success; it grants no blanket ORT 1.24.1 opset-26 support.
The native Eqx source and shared proof boundaries remain required at every
requested configuration.

The independent PRIMARY audit
`astra-eqx-dropout-review-v1/primary-native-array-audit.json`, SHA256
`c9449accf7caa3e20cd85128f443ea2cd706e6002fe86d8760b124acb6b7e47f`,
accepts the bounded saved evidence. It verifies **72 target pairs / 144 reports**
(45 training, 21 full-uniform and six inference cases), **540 report sample
observations**, and **132 actual saved target executions**. The saved arrays
come from an **additional explicit runtime session using the same settings**,
separate from the validator invocation. All 132 match the independent
oracle/native data and the validator output dtype/shape/payload hashes. The
six inference cases have reports without saved actual target arrays. These
counts neither describe 132 validator sessions nor establish universal
validator-session persistence. This independent array audit is PRIMARY-only;
the other two 100-case receipts retain their own actual test evidence.

**Original models and diagnostic status.** The four unchanged MLP0/1 recipes
(`raw-x64-off` and `raw-x64-on` for each ordinal) now pass artifact, schema and
lowering checks. The original factories, full parameters, source/options,
callbacks, skip flags and missing criteria remain. All four raw graph callbacks
remain **FAILED**, `criteria=None`, and runtime/numerics **UNVERIFIED**. These
raw recorder tests perform **no ORT execution**. In x64-on the model has strong
FP32 input together with FP64 stored weights/output and mixed-width dot work;
it is not an all-FP64-input model. The earlier diagnostic zero/scaling/finite
properties remain recorded, with **UNVERIFIED** diagnostic precision reports;
the new bitwise sample controls do not assign criteria to those diagnostics or
to the original MLP recipes. MLP2 and DINO state/replay capture remain unfinished
implementation/evidence work, not ONNX impossibility claims.

**Corrected combined epoch.** After the first affected failure and preliminary
quality failure below, ROOT applies exact test/support-only Dropout v4 and
replay-diagnostic v2 packets. The independent accepted review is
`astra-eqx-dropout-v4-and-diagnostic-v2-review/review.json`, SHA256
`cbbb792966e42aa3cb237c2282c1a4e7264397584cd396deadcceef46ff902e9`.
`eqx-dropout-v4-diagnostic-v2-integration-v1/applied.json` binds four exact
postimages and all **1,103 other held pins**, **1,107 total**. Production remains
the exact applied v3 implementation. The new 100-case owner's formatter changes
preserve its complete AST and original case identities.

The test diagnostic dispatch now recognizes the actual native
`jax.lax.stop_gradient_p` and sends its actual equation to mandatory
`validate_source_jaxpr`. Parameter, literal, dtype/shape/weakness, binding,
arity, effect and placement checks remain. This is no blanket family rule or
product-admission extension. Six positive and ten hostile controls now have
actual collection/execution in the corrected combined selection.

The actual ten-owner PRIMARY run,
`eqx-dropout-affected-v2-primary/receipt.json`, passes **680 unique cases**, zero
failures/errors/skips, in **314.130520676 seconds**, with source/Git state held.
Its XML SHA256 is
`136d0366cf872cdff56bcb66afe28c3db1c9556863e30f44e66deb84aca89bf8`.
Every earlier 285 affected identity and all original 100 new-owner identities
are present. The selection also includes the complete raw-replay and
native-capture owners. This is a separate source-bound execution, not a rewrite
of the earlier failed receipt. Passing an owner includes assertions preserving
failed/unverified target records; it does not accept every target or model.
The terminal SECONDARY receipt,
`eqx-dropout-affected-v2-secondary/receipt.json`, passes the **same 680 unique
identities**, zero failures/errors/skips, in **306.635090062 seconds**, with all
1,107 pins held. Its XML SHA256 is
`739612a3587abebc334b2410a8983e201936080ca37e3b87a596aab61c0a3cbd`.

**Actual direct-minimum failure and bounded counterfactual.** The same ten-owner
selection, `eqx-dropout-affected-v2-minimum/receipt.json`, records **675 PASS /
five FAIL of 680 unique identities**, zero errors/skips, in **236.489109010
seconds**, with all 1,107 pins held. Its XML SHA256 is
`45dca43465b8a6605bd9dbe4980ab433762f3306186eff7a54d586d89b769e13`.
The identities equal those of the two passing combined runs. Phases remain
**NOT_INSTRUMENTED**. The five failed identities are:

- `test_dropout_nested_default_and_literal_override_ignore_same_named_root_sample[False-nnx]`
- `test_dropout_nested_default_and_literal_override_ignore_same_named_root_sample[False-linen]`
- `test_dropout_nested_default_and_literal_override_ignore_same_named_root_sample[False-eqx]`
- `test_original_mlp_cpu_state_feedback_and_reset_replay[False-float32-linen]`
- `test_original_mlp_cpu_state_feedback_and_reset_replay[True-float32-linen]`

The first three fail during ORT session initialization; the final two fail
their unchanged exact sample assertions. The separate unchanged-`c686304c`
counterfactual, `eqx-c686-minimum-counterfactual-v1-minimum/receipt.json`,
requests the NNX/Linen nested cases and both Linen state-feedback cases. It
records **four FAIL of four**, zero errors/skips, in **23.638209049 seconds**,
with held sources and the same failure signatures. Its XML SHA256 is
`4d8e580662b4e2c88e569b36d325b06cf34edd486bd5d79166a488f024240bf1`.
The Eqx failed identity was not part of that old-source counterfactual; no
old-source Eqx outcome is inferred. The separate causal probe is pending at
this draft's cutoff. All five current failures remain open, without a waiver
or inferred repair. These reproductions supply provenance for four failures,
not a completed minimum gate. The earlier three-stack 100-case owner success
retains its own v3 epoch and does not establish a unified green minimum run.

**Bounded replay diagnostics and actual pre-native observations.** Applied
Packet A v2 enriches the existing unsupported function class/static-state
failure with descriptor-safe class/metaclass symbols, stable traversal path,
actual loaded module binding and current-Python callable/code fingerprint.
Full original names decide identity; shortened labels are never lookup keys.
The serialized diagnostic is capped at **32,768 ASCII JSON bytes**. Diagnostic
construction errors retain a constant unavailable marker and cannot replace
the original rejection. There is no class admission, adapter, factory/source
invocation, native computation or criteria change.

Packet A v1 remains rejected/unapplied. Independent reproductions showed a
foreign module-cache `__class__` property replacing the original error and a
3,000,237-byte unbounded diagnostic. V2 resolves those exact counterexamples;
only v2 received the accepted review and application above. The corrected
680-case run includes the full diagnostic owner, actual capture fixtures and
unchanged native-owner/scalar-alias/snapshot controls.

ROOT separately observes unchanged registered `MlpExample` ordinal 2 in both
original scenarios. The external
`eqx-mlp2-diagnostic-probe-v1-primary/receipt.json` records **two passing
diagnostic assertions**, zero failures/errors/skips, in **12.821239080 seconds**,
with held sources. These are pre-native class-state observations, not successful
model preparation or product acceptance. The exact observations are
`eqx-mlp2-diagnostic-observation-v1-primary/raw-x64-{off,on}/observation.json`.
Both identify actual class **`jax._src.traceback_util.UnfilteredStackTrace`**,
metaclass `builtins.type`, **exact** loaded defining-module binding and path
**`root/capture:mapped/global:UnfilteredStackTrace`**. The callable belongs to
`jax2onnx.plugins.examples.eqx.mlp`, code SHA256
`d4cad221e4275739e3652e4a7a671de914579d3b6340c2d5e46c7901f5589058`.

Each observation preserves the unchanged recipe's one-instantiation preparation,
original scenario/settings and **1,852-row / 3,704-cell** denominator, with all
**3,703 other cells retained by identity** in that observation. The preparation
still fails at instantiated-target-state; declared/concrete traces, eager,
fresh JIT and product admission are not attempted, native outcomes are empty,
and the raw callback is not executed. Artifact/schema/lowering, runtime and
numerics remain **UNVERIFIED**, with `criteria=None`. Future adapter selection
is planning only, not implemented or qualified. The optional original-MLP numerical packet remains
design-only. Its reviewed design authorizes external implementation work,
without an accepted numerical harness, provider-accuracy claim or native/target
results.

The separate external callable-preparation v4 refresh retains all three exact
v3 postimages and their ASTs against the current 1,107-file source epoch.
Its manifest is `callable-preparation-draft-v4/manifest.json`, SHA256
`595f9c0d2794a7d42ffbde98046bb93063455da9b9cdaf929935774cf1cccb55`.
Its **74 AST-estimated combinations** are not native collection/execution
counts. The packet is unapplied and native qualification is **NOT RUN**; the
concrete capture wrapper remains unimplemented. Existing NNX preparation and
actual GPT source files remain unchanged. These remain full fixed-source
implementation obligations. This source-capture gap is not an ONNX limitation.

**Retained failures and remaining gates.** The first actual seven-owner affected
PRIMARY run, `eqx-dropout-affected-v1-primary/receipt.json`, is **FAILED:
278 PASS / seven FAIL of 285 unique cases**, zero errors/skips, in
**181.941457284 seconds**, with source/Git state held. XML SHA256 is
`5851997b0224eb06a735d29d27b79d0670e35a9c989a337bc40321e6639b42f9`.
The seven failures are the existing Eqx positive-rate root/Function/If/Loop
cases and named default/True/False override cases. Each encounters the original
native `stop_gradient` inside its dynamic branch, outside the diagnostic
structural validator's admitted family. This failed epoch remains unchanged;
the corrected 680-case execution above belongs to its own later source epoch.

The preliminary `eqx-dropout-quality-v1/receipt.json` retains typing PASS
(57.825830175 seconds) followed by Black FAIL (1.907895301 seconds), with all
1,107 pins held; later checks were not run. The accepted AST-identical
formatter correction does not relabel that failed quality epoch as passing.

The later `eqx-dropout-quality-v2/receipt.json` records **all seven checks
PASS**: typing, actual-hook Black on eight changed Python paths, global Ruff,
reflection, builder, path markers and variable annotations. All **1,107 source/
configuration pins hold**. Strict docs/rendered links, normal commit hooks,
qualified bytes and commit checks remain pending. The current full PRIMARY
suite has **NOT RUN**. Quality results do not waive failed minimum checks.

| Current addition gate | Actual status at this documentation cutoff |
| --- | --- |
| Original exact v3 / new 100-case owner | Same 100 IDs PASS on three stacks in the earlier v3 epoch |
| Independent saved-array audit | Finite PRIMARY evidence accepted; 132 saved target executions in additional same-settings sessions |
| Applied v4 test proof / diagnostic v2 | Exact static review accepted; four applied postimages, production v3 unchanged |
| First affected seven-owner PRIMARY | FAILED: 278 PASS / seven FAIL of 285; retained |
| Corrected combined ten-owner PRIMARY / SECONDARY | Same 680 IDs PASS per stack; zero failures/errors/skips |
| Current direct-minimum ten-owner selection | FAILED: 675 PASS / five FAIL of the same 680 IDs; zero errors/skips |
| Unchanged c686 four-case minimum counterfactual | Four FAIL of four; Eqx failed identity not included; no waiver or repair |
| Minimum causal probe / repair / requalification | PENDING; all five current failures remain open |
| Actual MLP2 pre-native class-state observations | Two diagnostic assertions PASS; native outcomes empty, product/numerics unaccepted |
| Future callable preparation / capture wrapper / model numerics | External v4 exact v3; 74 AST estimates, native NOT RUN; wrapper unimplemented; numerics design-only |
| Current complete PRIMARY | NOT RUN |
| Preliminary quality | Typing PASS / Black FAIL; retained |
| Current source quality v2 | All seven checks PASS; 1,107 pins held |
| Strict docs / rendered links / normal hooks / qualified bytes / commit | PENDING |
| Full immutable 3,704-cell map / original-model numerics / B–F | OPEN |

Earlier six-path Dropout candidates v1 and v2 remain exact, rejected/unapplied and natively
unexecuted. Their patch SHA256 values are
`40f00fba888c8a3e7462f929f246198a738be652ab7e3ed21ad1bd3ea257d04d`
and `da08f7581fc9269790416c1ba7a1b71b863b6dba5a731389a99d179877c1db4e`.
V1 retained typed-key NumPy coercion, palindromic cache ordering and incomplete
mask-only random evidence; v2 retained an invented LEGACY typed-key-alias
smoke contract. Of those three six-path candidates, only v3 received static
acceptance and native execution. Its later test/support corrections have their
own accepted review and combined execution above. The separately rejected diagnostic Packet A v1 remains unapplied;
the accepted diagnostic v2 and its bounded actual observations are recorded
above without inheriting any v1 acceptance.

**Immutable product map.** The full original **1,852-row / 3,704-caller** map
actually started on detached immutable `c686304c`, separately from new-code
qualification. `eqx-dropout-map-pause-v1/{before-signal,after-child-stop}.json`
records the controlled pause for serial qualification: **70 ordinary terminal
cells, one incomplete `user_interrupted` attempt and 3,633 unstarted cells**.
The prior terminal hashes hold, the verified supervisor is gone and owned child
groups are empty. The interrupted cell is `issue18_fori_loop`, ordinal 0,
`raw-x64-off`, index 70. All original attempts and the complete denominator are
retained. Resume that same epoch with incomplete retry; never splice new-code
results into its cells. The map is product-only: its runtime/numerics remain
UNVERIFIED. Pausing it neither drops unfinished cells nor completes the goal.

The target still requires all existing functionality, fixed third-party
sources, original numerical criteria and full B–F acceptance through one
ordinary source-preserving export contract, without a permanent SOURCE/profile
selector or capability loss. Optional models, metadata-free anchors, original
callbacks, state/RNG/replay preparation and complete-model numerics remain
obligations. The [preceding committed checkpoint](#equinox-forwarding-shared-symbolic-contraction-and-output-axes)
retains its own failed full run and explicit corrected partitions below.

### Equinox forwarding, shared symbolic contraction and output axes

The preceding generalized-contraction/support-floor checkpoint is committed at
`7b70b22f`. The subsequent combined checkpoint is committed locally at
`c686304c6d9395be9e3df3dd3770c5a99ea1c2f5`. Its combined private epoch applies the separately frozen
Equinox original-call, shared native symbolic-contraction/transpose and exact
output-axis sample/report packets. Equinox forwarding preserves the captured
original `__call__`, arguments, callback behavior, native errors, stored model
parameters, aliases and state. Original five Linear rows and MLP rows 0/1 retain
their source/options/criteria; this does not claim all original MLPs pass.

One actual atomic `B`, derived from the unchanged positive leading-`B` input
contract, can appear at admitted intermediate and output axes. For example,
native weight-first `[64,128]` by `[B,128]` produces `[64,B]`, with an actual
transpose when the source asks for `[B,64]`. General dot normalization restores
the original ordered batch/free axes. Nonleading ports require exact physical
dtype/shape, alias and native symbolic-scope proof. Outputs with zero static
tails retain positive input `B` binding; this grants neither zero-tail symbolic
inputs nor `B=0`, and does not admit unrelated operation owners.

The corrective mixed dot route admits actual `(FP32,FP64)` and reversed
operands with a native FP64 result and absent/FP64 preferred type. Both installed
JAX CPU lowerings convert mixed nonalgorithm operands to that result dtype;
the shared lowerer validates original physical ports before a local Cast of
only the FP32 consumer value, then uses DOUBLE MatMul. Native precision enums
retain their reviewed meaning; explicit algorithms remain excluded. Original
input/storage/alias widths are unchanged. Scalar/outer/empty routes retain
their own exact native semantics, including signed zero and the positive-zero
sum identity. This adds no BF16/integer/complex arithmetic or
same-FP32/preferred-FP64 admission.

Both first combined installed-stack epochs remain **FAILED: 2,312 PASS /
12 FAIL of 2,324 unique cases**, zero errors/skips. Their actual receipts,
XML/logs and held sources remain in `eqx-symbolic-v1-{primary,secondary}/`.
The corrective tests preserve their original identities and numerical criteria.
They repair defining-module ownership for nested named calls and assert actual
serialized root-to-outer-to-child Function edges. An actual captured outer
body containing the registered child Function primitives still has no native
CPU MLIR translation for those carriers. Only that exact fresh-JIT exception
is recorded **NOT_ESTABLISHED**, tied to the actual definition and native trace.
Outer-body eager, both leaf-body eager/JIT and ordinary original-call
eager/warmed/fresh-JIT checks remain required before/after export without cache
clearing. Both corrected runs retain eight actual whole-carrier
**NOT_ESTABLISHED** receipts each. Passing observer tests do not turn those
receipts into successful captured-carrier JIT execution. This gap remains an implementation/evidence obligation, not an ONNX
or provider impossibility or a permanent target restriction.

The retained v2 31-owner selections pass the same **2,358 unique identities**
on both supported stacks, zero failures/errors/skips. PRIMARY completes in
**226.85684135 seconds**; SECONDARY in **242.541162873 seconds**. Both hold
the same **1,102 source/configuration files and Git state**. All original 2,324
identities are present and pass, including all twelve previously failed IDs;
only 34 cases from the new mixed-width owner are added. Phases are
**NOT_INSTRUMENTED**, rather than inferred from pytest cases.

PRIMARY is Python 3.12.9 / JAX and jaxlib 0.12.0 / Flax 0.12.10; SECONDARY is
Python 3.11.12 / JAX and jaxlib 0.10.2 / Flax 0.12.8. Both installed targets use
ONNX 1.23.2 / ORT 1.31.0. Their exact receipts are
`eqx-symbolic-v2-{primary,secondary}/`; `dual-focused-audit.json` and the
identity audit in `eqx-symbolic-integration-v2/` bind the unchanged old IDs and
the 34 additions. PRIMARY XML SHA256 is
`234371d22a41bdb95393fa157d0d7d97199884468cb2478f48bf962a4f337cf8`;
log SHA256 is
`752919ce9e46319152b9cf663065b932ce3deb0bcb5d6beb88dc54aefe309a71`.
SECONDARY XML SHA256 is
`4650955b6d190436952b1adbc57209587dfd1eab8b76cadd01c48f5a95d22f6e`;
log SHA256 is
`82088e17ed08632c849fa4c107748ab881634ab43325056bb9298755503968a0`.
The independent actual PRIMARY audit is
`astra-correction-review-v1/primary-v2-independent-audit.json`.

The actual `eqx-symbolic-quality-v1/` run passes Black on sixteen Python paths
and Ruff, but typing fails on a redundant `cast(int, axis)` among 559 checked
source files. The retained failure is corrected by removing that one typing
cast. `eqx-symbolic-typing-fix-v1/ast-runtime-identity-proof.json` verifies exact
AST equality after only that removal and stdlib `typing.cast` object identity;
the patch SHA256 is
`ffc778f4d1559fd1074954374626c97a4f0b8b703d447975d4b21aa5a4fff8a4`.
`source_profile.py` then has SHA256
`6da3b910b0c6b09e0826214e1ab45a456691abdb5e064de0e4dff6bcf0c71aaa`.
This is a distinct source epoch. The preceding dual v2 outcomes retain their
own bytes. The actual `eqx-symbolic-quality-v2/` run then passes all seven
quality checks: typing, pinned Black, Ruff and the four reflection/builder/
marker/annotation policies. All 1,102 source files hold. This is separate from
the seven normal commit hooks, which remain required checkpoint gates. Later functional packets create new source epochs; their final-source native and quality gates remain pending.

The direct-minimum capability preflight uses ONNX 1.20.1, ORT 1.24.1 and
Equinox 0.13.1. ONNX advertises schema maximum 25, yet the checker accepts the
actual opset-23/25/26 controls. ORT session creation passes at 23/25 and fails
at 26 with an explicit official maximum of 25. Checker success is therefore
not runtime capability or numerical acceptance. The actual preflight is
`eqx-symbolic-direct-minimum-opset-preflight-v1.json`.

The post-cleanup 49-owner minimum request collects **2,837 unique identities**.
The independently reviewed exact capability ledger excludes **145 positive
ORT-dependent opset-26 identities**, leaving **2,692 executed cases**. All 34
mixed-width cases, opset-23 counterparts, negative/metadata controls and the
ReferenceEvaluator-only TensorScatter opset-26 control remain. The initial
146-ID proposal incorrectly included that reference-only control and was
corrected before execution. The exact ledger is
`eqx-symbolic-direct-minimum-capability-ledger-v2.json`.

That actual minimum epoch is **FAILED: 2,665 PASS / 27 FAIL**, zero errors/skips,
in **308.906268658 seconds**, with all **1,102 source/configuration pins and Git
state held**. All 34 mixed-width cases and the retained reference-only control
pass. XML SHA256 is
`719e2f35d578432c226f9456247776b29b43083cd22d0a8d73b6b5293ed848f3`;
log SHA256 is
`91bf0fb245429ca087a14586e50c2979ee11a945ec9246acb2dac2e74afde171`.
`floor-v3-independent-failure-audit.json` preserves all exact IDs and traces:

| Cases | Observed minimum failure | Required disposition |
| ---: | --- | --- |
| 4 | Original BF16/FP16 iota rows request Range27 forms absent from ONNX 1.20.1's installed schema | Retain original source, opset and callbacks; any later exact schema-capability ledger has a distinct scope from the ORT ledger. |
| 2 | Mixed-role Function/cond artifacts require BitCast26, absent from that checker | Retain the checker failure separately from runtime session capability. |
| 12 | Default ALL creates `com.microsoft.FusedMatMul` DOUBLE; this CPU kernel accepts FLOAT only | Preserve same-artifact DISABLED acceptance and default-ALL runtime-load failure with the unchanged raising gate. |
| 4 | Subnormal controls meet an unrecorded runtime/framework pair before their strict mismatch assertions | Preserve the actual bit-2 target mismatch against native zero; adding the factual target observation cannot accept the numerical failure. |
| 2 | Full optimization loses `add_out_0` for empty FP32 geometry9-direct | Diagnose the original graph and shared static-empty Add representation; no Eqx source rewrite. |
| 3 | Original Eqx rows2/3/4 contain captured FP64 weights, FP32 input, x64OFF and native preferred/output FP32 | Preserve actual native semantics with a pre-dot local FP64-to-FP32 conversion and unchanged stored weights. |

No source, requested opset, criterion or declared minimum is changed to conceal
this failed epoch. It is not full 49-owner qualification and cannot be relabeled
a pass after later repairs or provider-evidence assertions.

ROOT then executes the frozen **40-observation empty-Add probe** on the two
original saved models, retaining all eight artifact/feed/native-reference hashes.
Both native references and all seven output ports stay exact. Original and
consumer-only Identity variants fail basic/extended/all while disabled validates;
first-Add Identity, both-Add Identities and typed-empty-initializer diagnostics
validate in all four optimization modes. The independent audit verifies **196
saved target arrays** against both native references. This supports a shared
static-empty representation; diagnostic graph variants are not mislabeled as
original-artifact acceptance or production qualification.

The separate **37-observation minimum artifact probe** retains **14 validated /
23 failed reports** across three representative FP64 and four subnormal
artifacts. Independent decoding verifies **52 matching target arrays and 40
retained bit-2 mismatch arrays**. For all three FP64 artifacts, default ALL fails
load after MatmulTransposeFusion creates an unsupported DOUBLE FusedMatMul;
DISABLED and the explicitly named optimizer-disable diagnostic validate. The
standard serialized source graphs are unchanged. The explicit diagnostic
configuration is not a new API, silent retry, source cast or default change.
For all four subnormal artifacts, unset or explicit `0` for
`session.set_denormal_as_zero` retains failed bitwise comparisons and raising
gates in both modes; explicit diagnostic `1` yields native positive zero.
Those diagnostic settings are carried in a sidecar because the unchanged raw
CpuSampleTarget report does not encode the injected option overlay. Raw reports
alone do not establish complete diagnostic target identity. The actual receipts
and independent artifact/terminal audits retain every failure and option.

The shared preferred-FP32 extension subsequently narrows only the FP64 dot
consumer before computing, after original physical checks and frozen native
inference. It adds no FP32 HIGH/HIGHEST or explicit algorithm admission; the
old native-FP64 precision gate is AST-exact. A K=1 rational witness in both
operand orders and both caller x64 contexts distinguishes pre-dot narrowing
from FP64 product followed by final output cast. Other finite coordinate cases
are observed samples, not a general no-rounding theorem.

Its actual-minimum selection passes **60 cases**, zero failures/errors/skips,
in **14.319799089 seconds**, on the **1,103-file epoch**: **46 new finite controls
and 14 unchanged raw Eqx/MLP outcome tests**. The three formerly source-rejected
Eqx recipe identities now meet their source product assertion. The raw outcomes
remain **ten callback failures and four profile exclusions**; runtime and
numerics are all **UNVERIFIED**. Their original absent criteria are not replaced
by the new finite controls. XML SHA256 is
`012f7235a100d9b048cb82671ff8e7fe1b5a5cd8df13521725b5e9582a16d80f`;
`preferred-fp32-floor-independent-audit.json` binds the result. This is bounded
minimum evidence, not a full affected or whole-model acceptance claim.

The separate shared static-empty Add repair reuses existing owner admission,
frozen original native Add inference, and exact physical checks on both operands
and the fresh output. An exact-result-shaped operand feeds a fresh Identity;
complementary singleton broadcasts use a typed ConstantOfShape. Only the output
Var is bound; inputs/parameters/aliases are never renamed, retyped or rebound.
The compatibility registry adds only the frozen native Add evaluator. Nonempty
and LEGACY bodies stay unchanged; zero-tail symbolic Add still fails its
existing owner gate, and dedicated complex Add retains its existing branch.
Empty packed-complex provider behavior remains unqualified, not a target-state
exclusion.

The first new-owner minimum epoch fails **before any test body: one collection
ERROR**, due to a nonexistent compatibility-module `jaxpr_as_fun` import. Its
receipt/XML/log remain in `eqx-empty-add-v1-floor-minimum/`. An import-only patch
uses the public `jax.extend.core` export verified in both installed JAX sources;
all test bodies and nonimport AST stay exact. The corrected actual-minimum owner
passes **all 38 cases**, zero failures/errors/skips, in **10.324057671 seconds**;
XML SHA256 is
`1a212b32f348f128b9c8c5543fe3e4fc3ebfe074ee6e5f47000f472664561f29`.
The separate two unchanged original Eqx geometry identities also pass, in
**15.040594230 seconds**. Their independent audit retains **40 finite geometry
sample reports**, including geometry9 under both caller contexts and both
optimization modes, all seven outputs exact without a session-config override.
These are distinct source receipts; none retroactively passes the failed
collection or minimum-v3 epochs.

The later shared test-evidence helper retains both complete original reports
for the same standard ONNX artifact and requested sample identities before
asserting anything. DISABLED must pass its actual raising gate. ALL either
passes its same gate or exactly reproduces the observed ORT 1.24.1/x86_64 CPU
Transpose/DOUBLE-MatMul optimization failure, with artifact/schema checks still
validated, load failure explicit and numerics UNVERIFIED. Its failure must still
raise `SampleAcceptanceError`. Constructor failure records sample zero only;
later requested ALL indices are recorded unexecuted and receive no acceptance.
Unexpected versions, stages, settings, errors, identities or source provenance
fail. No retry, injected optimizer option, dtype cast or production default
changes. A future target that actually validates is accepted through its real
raising gate rather than forced to reproduce a permanent error.

The helper's 20 actual-minimum controls pass as tests, including measured
native/coordinate/target baselines and hostile classifier cases. Synthetic
classifier records are not extra measured provider outcomes. The four original
FP64 owners use the same shared assertion; the existing strict subnormal branch
adds only the factual JAX 0.10.2 / ORT 1.24.1 pair. Bit-2 comparisons remain FAILED
and their raising gates remain required in both modes. These negative witnesses
account for failures; they never establish numerical acceptance of those failed
target settings. The 27-failure minimum epoch and both diagnostic probes remain
unchanged historical evidence.

The subsequent SECONDARY 35-selector run is **FAILED: 2,609 PASS / one FAIL of
2,610 unique cases**, zero errors/skips, in **683.137202409 seconds**, holding
all **1,106 source/configuration pins**. The sole failure is the old unselected-
native-evaluator fixture still using Add after Add became explicitly selected.
A separate test-only correction uses genuinely unselected Mul while retaining
that test's identity and guards. The actual whole **83-case native-contract
owner then passes**, zero failures/errors/skips, in **11.177090748 seconds**.
Production, configuration, versions and worker settings match between epochs;
only the reviewed test file changes. The accepted affected-coverage partition
is **2,527 unchanged-other-owner passes plus 83 corrected-owner passes**,
disjoint and complete over the original 2,610 IDs. It is explicitly two epochs,
not a unified green 35-selector run. The independent
`direct-minimum-ledger-v5-and-secondary-partition-independent-audit.json`
binds this distinction. Original failed-run XML SHA256 is
`a644006a5b4ae37ebc4df511fa8ee48b6a92f264b97f4a6324946b50c9bf7b0d`;
replacement-owner XML SHA256 is
`981209e6c708e6f1b2c71dff14925356e72b1571d9c55fdc65ab8d6b7e61850b`.

The final-v5 direct-minimum collection contains **3,089 identities across 53
selectors / 52 files**. The exact ledger has **181 exclusions**: the earlier
145 positive ORT-opset26 cases, 30 newly selected existing complex ORT-opset26
cases, and the six independently proved installed-schema failures. The
ReferenceEvaluator-only opset26 control and all 21 non-schema failures from
the earlier minimum epoch remain in execution, along with the 46 preferred-FP32,
38 empty-Add and 20 target-helper controls. No source, requested opset, original
criterion or runtime minimum changes to avoid those capability boundaries.

That exact **2,908-case subset passes**, with **zero failures/errors/skips**,
in **461.287419929 seconds**, holding all **1,106 source/configuration pins and
Git state**. XML SHA256 is
`5ad7b463d7603c346f79bb90fd8d4f2b412df6e1d3d0533f335cb4b8b58c468e`;
log SHA256 is
`b0f92853955bb0709c4fc194a2287c5d84e9429ca622aa46e9eab8fab514a25c`.
This is a capability-scoped actual direct-minimum result, not full 53-selector
or all-functionality acceptance. Passing negative target tests do not change
the failed FP64 ALL or subnormal sample reports into successful samples.

Independent final-minimum identity and raw-report audits bind the exact result.
The report audit verifies **680 actual DISABLED/ALL report pairs**: 660 ordinary
owner pairs and 20 helper-baseline pairs. **647 pairs validate in both modes**;
**33 retain default-ALL runtime-load failures** (14 ordinary and 19 helper
baselines). Every negative keeps numerics UNVERIFIED, actual providers empty
after constructor failure and the unchanged raising gate. No optimizer/denormal
overlay is applied. These are finite report counts, not distinct-model or
universal coverage claims. The four original subnormal cases retain **eight
FAILED bitwise reports**, native zero versus target bit-2, with original disabled
inputs, references, outputs and criteria unchanged. Four matrix-admission
negative owners retain digest/provenance reports without a separate saved
`source.onnx`; there is no universal raw-artifact/array persistence claim.

The earlier quality-v3 epoch passes typing, pinned Black/Ruff and four policies,
holding **1,104 pins** before the target helper and later test-only correction.
Typing takes **99.120626042 seconds** and Black checks nineteen changed Python
paths. That earlier receipt does not qualify later test changes.

The subsequent quality-v6 epoch is **FAILED**: typing on 559 source files,
pinned Black on 27 changed Python paths, Ruff, reflection and builder checks
pass, but the path-marker check fails on the two new target-evidence helper/test
files. Annotations do not execute after that failure. Its receipt and marker
log remain retained. A two-file comment-only patch, SHA256
`4e3fccbb717e60a2a16e952fd44aaa7526dc6e4b10b672d9f5434f3399dd3698`,
adds each required file-path header and blank line. All remaining bytes and
both complete module ASTs are exact; production and the other 1,104 input pins
remain unchanged. The new 1,106-pin comment epoch is distinct from the actual
native execution epochs above. The independent marker audit also verifies
compiled structure/constants excluding locations; no native repeat is claimed
for the changed comment bytes.

Actual quality-v7 then passes **all seven checks**, holding all **1,106 pins**:
typing on **559 source files** in **58.924235323 seconds**, pinned Black on
**27 changed Python paths**, Ruff, reflection, builder, path markers and
annotations. Receipt SHA256 is
`116ced393e599d64d49cf2b866a29e3a3296fcbecc73a27997a6ccf0faf8007d`.
This quality result is separate from the seven actual normal commit hooks
subsequently recorded below. The five-document update occurs after those held
native and quality epochs; no unchanged-all-inputs claim extends across that
documentation update.

The subsequent complete PRIMARY run is **FAILED: 12,410 PASS / six FAIL of
12,416 unique cases**, zero errors/skips, in **1,770.503410781 seconds**. All
**1,106 source/configuration pins and Git state hold** during that run. XML
SHA256 is
`d6aaa658b1802709cec4af85d3112cd97732596dc5c2fbe04eebaf9d48d75b95`;
log SHA256 is
`4155d4587ab3ee896b79bc49efe4c8f2a042b3b50a2e47d967b7af66c31d3c09`.
The complete six-trace failure ledger has SHA256
`909bba9b076f6c3dca6721fb4b051342923b0d6b457a6004d0d5d87d701a4bfd`.
This original full run remains failed regardless of later corrective results.

The six failures belong to four stale test-control owners. The mixed-width
FP64-preparation negative now describes a natively valid mixed-FP64-compute
route; its correction must test a genuinely contradictory FP32 output while
retaining the original ID and guards. One NNX-reshape and two Function-context
checks retain the former all-FP64-only diagnostic wording. Two Scan controls
still demand the shape inside a diagnostic now emitted earlier by the owner
preflight. The actual Scan detail is **"nonleading or zero-tail symbolic layout
has no audited owner"**; its profile exclusion and explicit unadmitted shape
remain unchanged. No production, reference, criterion or owner admission is
changed to repair these controls.

The original full XML contains **190 cases across those four owners**
(92 FP64-preparation / 47 NNX-reshape / 35 Function-context / 16 Scan) and
**12,226 other-owner cases, all PASS**. The reviewed four-test-file-only patch
has SHA256
`b644430a1efae91baf41701a7c6f13f5ce2e1ab825df35c55545fe36c37c2de7`;
production, configuration, helper bodies, test identities, references and
numerical criteria are unchanged. The final v6 source inventory contains
**1,106 exact pins** and differs from the full-v5 snapshot only at those four
test files.

The corrected four whole owners pass the exact same **190 identities**, zero
failures/errors/skips, in each of three actual executions:

| Execution | Actual duration | XML SHA256 |
| --- | --- | --- |
| PRIMARY | 32.745338983 seconds | `929de17ae764a62829b0a6db1a941fd12cc7a9c7fb991b988f670df13e4b1b51` |
| SECONDARY | 32.196797974 seconds | `5b1db7472ad87bb162da3c508124554b00a2af50bf8b3c06fb93e8f6e7b32677` |
| Actual direct-minimum supplement | 28.607686241 seconds | `6cd1ccbc23f062b613f839e778b0c18631c742390c0f38f6dbc6907e0e5f8e22` |

All 1,106 source/configuration pins and Git state hold within each execution.
The exact PRIMARY coverage partition is **12,226 unchanged-other-owner passes
from the original full run plus 190 corrected-owner passes**. The original
12,410-PASS / six-FAIL full receipt remains failed. SECONDARY evidence is the
earlier **2,527 unchanged-owner passes plus 83 corrected-native-contract-owner
passes**, supplemented by the new **190 whole-owner passes**. All four new
owners are absent from the earlier 35-selector SECONDARY and 53-selector
minimum selections. The actual minimum is the earlier **2,908-case subset with
181 exact capability exclusions**, plus this separate **190-case supplement**;
its original source epoch and ledger are retained. These explicit partitions
are not unified green full-suite, affected-suite or combined-minimum reruns.

The original full terminal audit, ROOT exact partition receipt and independent
three-lane identity/source/partition audit bind these distinctions. The audit
also keeps the actual provider-negative sample reports separate from passing
test executions. These finite checkpoint results do not establish whole-model,
full current-stack B–F or complete-product-map acceptance.

| Gate | Actual status |
| --- | --- |
| First combined PRIMARY / SECONDARY | Each FAILED: 2,312 PASS / 12 FAIL of 2,324; retained unchanged |
| Retained pre-typing v2 31-owner PRIMARY / SECONDARY | Each 2,358 PASS; same IDs and 1,102 pins |
| Post-typing minimum-v3 capability subset | FAILED: 2,665 PASS / 27 FAIL of 2,692; 145 declared capability exclusions |
| Saved-artifact diagnostics | Actual 40 empty-Add and 37 broader observations; diagnostic completion only, failed reports retained |
| Preferred-FP32 / corrected empty-Add minimum follow-ups | 60 PASS / 38 PASS, plus 2 unchanged original Eqx geometry cases PASS in separate receipts |
| Empty-Add first new-owner epoch | One collection ERROR before test bodies; retained |
| SECONDARY first 35-selector epoch | FAILED: 2,609 PASS / one FAIL of 2,610; retained |
| SECONDARY affected coverage | 2,527 unchanged-owner PASS + 83 corrected-owner PASS across two explicit epochs |
| Final-v5 capability-scoped direct-minimum | 2,908 PASS / zero failures/errors/skips; 181 exact capability exclusions; separate v6 supplement below |
| PRIMARY full-v5 original | FAILED: 12,410 PASS / six FAIL of 12,416; all original evidence retained |
| Corrected four whole owners | Same 190 identities PASS on PRIMARY, SECONDARY and actual direct minima; three separate receipts |
| PRIMARY coverage partition | 12,226 unchanged other-owner PASS + 190 corrected-owner PASS across two epochs; original failed full retained |
| SECONDARY affected partition plus supplement | 2,527 unchanged-owner PASS + 83 corrected-owner PASS + separate 190-case/four-owner supplement |
| Direct-minimum subset plus supplement | 2,908 PASS with exact 181 exclusions + separate 190 PASS supplement; no unified rerun |
| Earlier quality-v3 | Seven quality checks PASS on its pre-helper 1,104-pin epoch |
| Quality-v6 | FAILED: five checks PASS, path markers FAIL, annotations not executed; exact two-file comment-only repair |
| Quality-v7 | Seven checks PASS on its 1,106-pin header-comment epoch; 27 changed Python paths, typing on 559 source files |
| Seven normal hooks / held bytes / local commit | Subsequently completed at `c686304c`; exact receipts below; no unified green full-run inferred |

The subsequent actual normal commit succeeds at
`c686304c6d9395be9e3df3dd3770c5a99ea1c2f5`, parent `7b70b22f`, in
81.300028776 seconds, without skipped hooks or `--no-verify`.
`eqx-final-integration-v8/commit-receipt.json`, SHA256
`295362e22a7a5a274e09e97051d65ef244fd49c29f2cc21988f456769e818e59`,
records all **1,106 qualified input bytes held**, **27 Python / five reviewed
documentation paths**, all **32 changed committed blobs exact**, and a clean
worktree. The independent
`astra-correction-review-v1/checkpoint-c686304c-independent-commit-audit.json`
verifies all seven actual hooks: Black, Ruff, reflection, mypy, file-path
markers, IR-builder policy and module annotations. It separately accounts for
existing `.python-version` CRLF-to-LF Git input normalization; all 942 Python
inputs are exact. This closes that local finite checkpoint's normal-commit
and byte-binding gates. It does not turn the retained failed 12,416-case full
run into a unified green run or complete original-model numerics, the full
3,704-cell product map, release or B–F acceptance.

ROOT verified archives for both failed v1 and passing v2 installed-stack RAM
trees, preserving their sources. The v2 archive-only receipts bind tar SHA256
`4c8483334e9b0dba9308c8d0049927b00d01524b1958fd386f8d892c29d47d81`
(PRIMARY) and
`8b2573f639db68d14e96eb1093c1d5c24db4cce8b61258c56fabd217f2ff2fc0`
(SECONDARY), under the owned Windows `ram-artifact-archives/` evidence root.
These contain RAM/basetemp evidence only; generated cwd-relative `onnx/`
outputs are outside that scope. At archival, protected-process visibility was
incomplete; release requires a complete fresh visibility check. The RAM sources
remain retained, and no release is claimed.

New finite mixed-width coordinate samples predeclare bitwise dtype/shape/payload
equality and retain original eager/fresh-JIT arrays, activated traces, serialized
artifacts and actual runtime payloads under both CPU optimization modes. At
the retained v2 epoch, the shared symbolic observer retains native arrays and
sample reports but does not itself persist target output arrays. The new
preferred-FP32 and static-empty Add witnesses also save actual target arrays.
These sample controls do not assign criteria to original raw recipes or certify whole models. The
complete 1,852-row/3,704-caller map, optional models, metadata-free anchors,
original callbacks, full numerical requirements, general fixed-source/state/RNG
preparation, scientific FP64 and all B–F existing-functionality gates remain
open. Public behavior stays LEGACY; the standalone target remains one normal
source-preserving export contract across the existing functionality and fixed
third-party sources, without a permanent mode/profile selector or capability
loss.

### Generalized native contractions and product-map smoke

The shared contraction plan preserves ordered contracting/batch-axis pairs and
native free-axis order through typed Transpose/Reshape consumers and MatMul.
It covers the reviewed static same-width FP32/FP64 geometry and explicit
FP16-input/preferred-FP32-result boundary, including scalar, outer, batch and
empty cases. No-reduction products preserve signed zero; empty reductions use
the additive identity. The established positive tensor-by-matrix path remains
one MatMul. Native primitive identity, frozen abstract evaluation, aval/context,
result and physical representation checks remain mandatory. Oversized individual
ONNX dimensions and unfinished grouped intermediates have distinct categories;
an alternate strategy remains work, not a claim of mathematical impossibility.

SOURCE LinearGeneral/DenseGeneral and JAX/NNX/Linen einsum forward the captured
original call with exact positional and keyword arguments. Legacy bodies are
unchanged. Actual callback hooks, stored parameters, bias, consumer dtypes,
weakness, native errors and restored settings remain checked. Other dtypes,
explicit algorithms and broader symbolic contractions remain implementation or
qualification obligations rather than permanent exclusions.

The first applied v2 selection completed on all three stacks with **993 unique
cases: 988 PASS / five FAIL / zero ERROR or SKIP**. Four named tests failed
before export because their recorder compared a native dot directly with the
intentional named-function carrier. The fifth expected success after corrupting
`multiple_results`, which the frozen original native evaluator itself rejects.
The corrected tests retain their IDs and sources: they observe the actual
function-body trace paired with its exported definition, verify call context,
JIT parameters, original kernel/aliases/ports and native dot semantics, and
compare both native body roles plus whole-model outputs. Mutable abstract-eval
replacement cannot forge validation; corrupted result convention fails closed.
No production evaluator is copied or reconstructed. The three v2 failed epochs
remain in `generalized-contraction-v2-{minimum,secondary,primary}/`.

The first typing run also failed on three shape-annotation errors. Distinct
normalized `tuple[int, ...]` locals correct those errors without changing runtime
validation, arithmetic or branch order; the whole production AST is identical
after normalizing only those local names/annotations. Actual typing now passes
on **559 source files**; the **17 RNG factories** are an inventory count.

The corrected v3 selection passes the same **993 unique identities on PRIMARY,
SECONDARY and MINIMUM**, with **zero failures/errors/skips** and held sources/Git. The
new generalized-contraction owner contributes **376** cases. All five failed IDs
and all earlier 993 identities are retained. MINIMUM completes in **55.405045
seconds**, SECONDARY in **43.990266 seconds** and PRIMARY in **43.215915 seconds**.
On MINIMUM all **1,098 inputs**,
including **934 Python files**, match the run. Its four named models produce
**16 validated bitwise sample reports** (two native roles and both runtime
optimization settings). Saved native/oracle payloads match; actual captured-body
outputs match the original native outputs, and all four output/reference evidence
records agree in every sample. The original sources, bounds and parameter bytes
are unchanged.

The **22 original dot-row/caller observations on MINIMUM** retain **18 FAILED
raw callbacks, two PASSED callbacks and two profile-validation outcomes whose
callbacks did not execute**. All retain `criteria=None`; runtime and numerics
remain **UNVERIFIED**, and this replay performs no ORT execution. Passing these
recorder tests does not accept the failed callbacks or close the eleven-row
family. Pinned Black/Ruff on all twelve changed Python paths and all four
policies pass; receipts are in `generalized-contraction-style-policy-v3/`.

The subsequent complete PRIMARY v3 run is **FAILED: 11,992 PASS / 69 FAIL
of 12,061 unique cases**, zero errors/skips, **691.290155 seconds**. All 11,685
preceding IDs, all 993 focused IDs and all 1,098 source/config pins were
retained; the only 376 additions are the generalized-contraction owner.
Both v4 affected runs also remain **FAILED: 1,419 PASS / 21 FAIL of 1,440**,
zero errors/skips, with identical IDs and held sources.

The v5 fixture correction retains each intended corruption using a fresh
native annotation on the same test variable and checks the specific unknown
physical-dimension error. It introduces no production mutable proxy, inference
replacement, skip or changed numerical bound. The same **1,443 affected cases
pass on SECONDARY and PRIMARY**, with zero failures/errors/skips: all 1,440
previous identities plus the three minimum-installer tests. Sources and Git
state hold within both runs; all 934 Python inputs still match after the
separately reviewed lock update.

The completed v5 PRIMARY full run passes **12,061 unique cases**, with zero
failures, errors or skips, in **660.279726 seconds**. Every earlier v3 full
identity and all 1,443 affected identities are present. Its 1,098 source/config
pins matched the held source checkpoint before this five-document status
update; all 934 Python inputs remain unchanged. This installed
PRIMARY environment is Python 3.12.9 / JAX and jaxlib 0.12.0 / Flax 0.12.10;
it is distinct from Poetry's Python 3.12 lock selection.

The original SECONDARY full run was interrupted by the VM restart. Its log
survives, but no terminal receipt or XML exists, so it has no passing result.
The new complete SECONDARY run finishes with **12,054 PASS / two SKIP of
12,056 unique cases**, zero failures/errors, in **1,836.001337 seconds**.
It uses Python 3.11.12 / JAX and jaxlib 0.10.2 / Flax 0.12.8 and exactly the
same 1,098 pinned source/config files as PRIMARY. All 1,443 affected identities
are present and pass. The two skips are the absent optional Keras integration
and JAX's absent `one_minus_square` primitive. Five PRIMARY-only identities
are three feature-gated `no_collectives` context cases and two generated
`one_minus_square` cases; no other identity is missing. These conditional
cases and skips are explicit limits of this installed-stack result.

The separate fresh Python 3.11 all-direct-minimum lane verifies all fifteen
runtime dependencies at their declared lower bounds, including ORT 1.24.1.
Its first actual sixteen-owner CI replay is **FAILED: 464 PASS / one FAIL of
465 cases**, zero errors/skips: the isolated setup omitted the declared
`mypy==2.3.1` development tool required by one typing test. This is a test-tool
setup failure; that guard did not execute its intended typing check. The
failed result is retained without changing source, criteria or runtime floors.
After installing the declared tool, the unchanged local sixteen-owner replay
passes **all 465 unique cases**, zero failures/errors/skips, in **104.421656
seconds**. It retains exactly the failed epoch's 465 identities and the same
1,098 original source/config and 960 execution-Python file hashes. This is a
local replay of the CI selection at all fifteen direct runtime minima, not a
hosted CI run or complete supported-functionality/B–F acceptance.

Checkpoint acceptance requires final quality, strict docs, all seven normal
hooks and qualified-byte verification. Full existing-functionality and B–F
acceptance remain separate obligations.

The failed full receipt/XML/log are in `generalized-contraction-full-v3-primary/`.
XML SHA256: `a0b7bdb23520631d460157ff0a206d211cca9425830623606ccbe5cfc1627e2c`;
log SHA256: `68419be50b23688eb396833185c44df92a47dab23977de947e1d705aad65cc06`.
The 69 failures exposed stale source fingerprints and native/context fixtures,
plus obsolete rejection expectations. Source review confirms all eleven
registration recipes, the NNX Linear module and established direct MatMul
lowerer bodies are unchanged. The source and dependent descriptor pins were
updated without changing counterpart topology/parameter-byte/status guards.
The v4 failures comprise 18 frozen-aval writes and three over-specific physical
error expectations; all original failures remain in their separate directories.
Strict docs, all seven normal hooks and qualified-byte verification remain
checkpoint gates. Phase counts are **NOT_INSTRUMENTED**.

The v5 affected receipts are `generalized-contraction-v5-floor-secondary/`
(**48.778984 seconds**) and `generalized-contraction-v5-floor-primary/`
(**46.197382 seconds**). Independent XML identity comparison retains every
prior 1,440 ID and exactly the three installer tests, with the same 1,443-ID
set on both stacks. The completed PRIMARY full evidence is in
`generalized-contraction-v5-floor-full-primary/`: XML SHA256
`f0f353122866fc1980512302eb9a41f300c9b799094241e8e0e68751559e9ee7`, log SHA256
`eb48896ca553c223e81b6e48807720a8fa7f6c0a31cddacedb7d612c7e6d3755`.
The interrupted `generalized-contraction-v5-floor-full-secondary/` retains
its inputs, version record, launch and partial log; no terminal result exists.
The independent restart audit verified receipt hashes, pre-documentation source pins,
all 12,061 historical PRIMARY full identities and the equal 1,443 focused ID sets.
The completed new SECONDARY evidence is in
`generalized-contraction-v6-serial-recovery-secondary/`: XML SHA256
`cde151623348d743ae1c243d61b34dfdcebac259c9934f647ed7c7b7e3eeeca7`, log SHA256
`e69171e1b9e539541cf0f82c6635d9604db020b3306bc96a28955a9853b03020`.
`astra-secondary-full-terminal-audit-v1.json` independently checks the actual
12,056 identities, both focused sets, all 1,098 pins and the source-based
conditional reasons. No skipped or absent case is counted as a pass.

The authorized support declarations were checked in an unpublished wheel;
its retained METADATA bytes match `support-floor-wheel-metadata-v1.json`. Both refreshed locks pass their
native consistency checks. UV retains all 66 package rows; Poetry retains all
188 previous package versions and adds 17 rows confined to the optional GPU
group. The corrected CUDA12 extra names an actual JAX extra; this is resolver
evidence, not GPU runtime qualification. Existing framework versions and
Python markers remain unchanged in each lock. The ml-dtypes and optree marker
normalizations preserve their effective dependency minima. The two lock audits
are `support-floor-{poetry,uv}-lock-audit-v1.json`. The exact framework-floor
stack remains distinct from all-direct-dependency-minimum installation.

The first fresh Python 3.11 all-direct-minimum installation failed in pip with
`OSError: [Errno 12] Cannot allocate memory`; no native compatibility tests ran.
This is an installation resource failure, not evidence of incompatible minimum
versions. The failed preparation receipt/log and the separately pinned source
snapshot remain available. The fresh Linux environment then installs all fifteen
direct runtime minima and passes `pip check`; origins resolve to its own Python
3.11 environment and isolated source. Test generation succeeds inside that
snapshot. Its `direct-minimum-ci-py311-v2/` replay records **464 PASS / one FAIL
of 465**, zero errors/skips, in **104.723806 seconds**, holding all 1,098 original
source/config files and all 960 execution Python files, including generated
owners. The sole failure is `No module named mypy` from the typing guard's
subprocess; the declaration already requires `mypy==2.3.1`. XML SHA256 is
`c16039e6f0e95d0f0d5befd026d60949626e4d417d91a25ec8a8a279c57414a1`;
log SHA256 is `3d20f674c63c34e19c9a3f2217ace93d5259cd504de169bb691813078bedb11b`.
The failed epoch and its model archive remain retained. Installing the declared
development tool preserves all fifteen runtime minima and `pip check` passes.
The unchanged sixteen-owner replay in `direct-minimum-ci-py311-v3/` then passes
**all 465 unique cases**, zero failures/errors/skips, in **104.421656 seconds**.
Its exact ID set and all 1,098 original/960 execution-Python hashes match v2;
no source, test body or numerical criterion changed. The actual versions again
verify all fifteen direct runtime minima, isolated origins, Python 3.11 and
JAX x64; `mypy==2.3.1` is present. XML SHA256 is
`bce02fa0de76ae85bf117961dee7c3234aeb3d72a35cb9639d32e71ad709ce7d`;
log SHA256 is `58e428077b36efffbdaa2fd04c43db249acd3c2ada9e15bb819f293402aa32a3`.
`astra-direct-minimum-v{2-failure,3-terminal}-audit.json` independently verifies
both raw outcomes. This is the local exact CI-owner replay, not a GitHub CI
result, a full minimum-environment suite or whole-migration acceptance.

Both minimum replay trees are durably archived, including each run's 227 ONNX
models in its own `onnx/` directory, basetemp outputs and original evidence.
Each tar is 6,922,240 bytes. The retained failed-v2 tar SHA256 is
`7efe1d61b26bedb2d39aa6dcb8bc1eaadbe467515c961766cef03fba873ab8be`;
the passing-v3 tar SHA256 is
`be77fab3814a05b784cd78423752f0023cb674afdb746bdef2f011eb025253ce`.
Their separate bundles are `ram-artifact-archives/direct-minimum-ci-py311-v{2,3}/`
under the same owned Windows evidence root. Neither RAM tree is released.

The restart cleared all remaining `/dev/shm` trees, including the contraction
pytest model outputs and the wheel itself. The durable test receipts, XML/logs,
source hashes and wheel METADATA survive; they substantiate the recorded runs,
but do not make those lost model or wheel bytes available for replay. Earlier
explicitly archived BF16 artifacts remain separate. The new SECONDARY run's
RAM/basetemp artifacts are durably archived under the owned Windows evidence
root in `ram-artifact-archives/generalized-contraction-v6-serial-recovery-secondary/`.
Its 1,926,748,160-byte tar has SHA256
`859bdcf000cab6269854986e41a6997fa056edeba1638221e4f9f9358ab469cb`; its manifest
has SHA256 `2b4bba4620d4b33b2ba20ca55807a9aad1c175c0eb0f01a7b858b5a82ba38dda`.
ROOT verified all 13,945 entries, exact original evidence and the unchanged
source inventory. The archive contains 1,588 ONNX files and two `.onnx.data`
files under `tmp/`, plus cache files. It does not contain the cwd-relative
`onnx/` tree of generated models. The existing MAIN `onnx/` directory is a
mixed-age observation, with no unique SECONDARY attribution. It is now preserved
separately in the host `observed-generated-models-v2/` bundle: 3,754 entries,
including 3,710 ONNX files and fourteen `.onnx.data` files. ROOT verified every
archive member, the unchanged source inventory and all 1,098 repository pins
before this documentation update.
The 6,394,439,680-byte tar has SHA256
`0996a1d5918487c7ee801335a7b3e72e509fc1b53e9bcc913d76fce366953128`;
its manifest SHA256 is
`eba78fd7e72d560254e1a261686aae5d700ed2c5bbfa2d7809e3fd4bdc1e1d70`.
The source tree remains in place. This observed snapshot does not establish
which full run produced each model; lost pre-restart artifacts are not recovered.

These archives record incomplete visibility into the protected owned `ssh-agent`
process. Archive-only copying was explicitly authorized; the RAM source remains
retained and strict release stays blocked by that visibility gap. No closed-tree
release is claimed. Fresh minimum-lane generated models are routed to each run's
owned RAM directory, so their model files and basetemp evidence are archived
together before rerun. `astra-secondary-archive-scope-audit-v1.json` records the
precise full-run archive scope; lost pre-restart RAM bytes remain lost.

Receipts, XML/logs and typing failures/passes are under
`/tmp/jax2onnx-restart-checkpoint-20261021/`, in
`generalized-contraction-v3-{minimum,secondary,primary}/` and
`generalized-contraction-typing-v{2,3}.log`. MINIMUM XML SHA256 is
`b37eca64524c863d25eb6316041a3caa84013e9c53c396c7edc2894ac0489896`;
log SHA256 is
`d0c657dfa48daebcb9fe1a92152357550ef6be14c9fe88d623c29fcf1a196675`.

The separate external coordinator calls the unchanged full-plan product API,
retaining **1,852 rows / 3,704 caller cells**, all **50 optional rows** and
**104 separate metadata-free anchors**. One fresh process per cell, bounded
resources, pinned imports/sources and durable attempt records separate ordinary
probe outcomes from interruptions. Sixteen synthetic tests precede actual smoke
on the immutable platform commit `b7eb923a`.

That smoke completes cells 0/1, then exactly cell 2 on resume. All **38 prior
files** remain byte-identical; all **54 published capture-file hashes** and
**1,097 frozen source/config pins** hold. Three attempts produce three ordinary
SOURCE profile exclusions at **`eqx.nn.linear`**, leaving **3,701 cells not
started**. Original eager/JIT/traces and model state are captured and state is
unchanged. The FP32 training native roles differ in bits, and those results are
retained. No artifact, graph callback, runtime or numerical acceptance is claimed;
raw criteria remain missing. Records live in
`/mnt/c/Users/enpasos/AppData/Local/jax2onnx/precision-evidence/20261021/source-product-platform-smoke-v1/`.
This is operational smoke qualification, not completion of the product map.

The three independently frozen packets were prepared as
`eqx-linear-source-draft-v2`, `shared-symbolic-contraction-draft-v3` and the
incremental `symbolic-output-axis-boundary-draft-v2`. Their subsequent applied
epoch, retained failures and corrective qualification belong to the separate
[Equinox/shared-symbolic record](#equinox-forwarding-shared-symbolic-contraction-and-output-axes)
above. Run a fresh full-map epoch only on a newly qualified immutable snapshot;
do not combine observations from unlike source epochs. These finite stages
neither establish whole-model numerical acceptance nor define a permanent
target restriction. Original MLP models, parameters, options and criteria
remain fixed. Complete
runtime/numerical criteria, original callbacks, optional models, metadata-free
anchors, fixed-source workflows and supported-floor/B–F acceptance remain open.

### BF16 elementwise feasibility observations

A separate frozen diagnostic completed on PRIMARY JAX/JAXLIB 0.12.0 and
ORT 1.31.0 CPU with x64 disabled, using both disabled and full optimization.
All seven declared models pass full ONNX schema checking. Its fourteen
observations keep runtime-load failures separate from executed comparisons:

| Frozen diagnostic branch | Actual result in each optimization mode |
| --- | --- |
| Direct BF16 Neg, Add and Add/Add/Neg chain | All three fail at session creation because this CPU provider has no matching BF16 kernels; no execution occurs. |
| FP32 Neg with BF16 castback | Physical BF16 input/output executes once; 252 of 65,536 words differ from each original native role, all involving NaN words. |
| FP32 Add with BF16 castback | Physical BF16 input/output executes once; 69,711 of 692,800 declared pairs differ, including 69,280 non-NaN differences. |
| Chain with BF16 rounding before both consumers | Each of its three 16-word outputs differs at three positions from each native role. |
| Delayed-rounding chain control | The second output differs at ten positions; the other two still differ at three each. |

Original eager and fresh-JIT outputs agree on all declared inputs. The saved
word comparisons retain the experiment's predeclared bitwise criterion,
including its NaN-word differences; they do not impose that criterion on raw
rows with `criteria=None`. The finite input catalogue, this provider/version
and the tested representations establish neither BF16 arithmetic
admission nor a general impossibility result. No production fallback or public
behavior changes. A broader numerical contract still requires its own prior
criterion and evidence.

The diagnostic exits zero because observation collection completed; its failed
branch/reference verdicts remain failed. The owned host bundle
`bf16-elementwise-primary-v1/` contains all 50 hashed artifacts, 8,978,921 bytes,
plus receipt SHA256
`0a48e382f4c94a539dccbb32b949b2004a28d4e7c1f136822f6ace4b05306cbd`.
`astra-bf16-elementwise-primary-audit-v1.json` independently rehashes every
artifact, reproduces every saved word/group comparison and verifies the held
source/probe/input pins. Context restoration passes; peak RSS is 337,624 KiB.
The earlier resource-guarded attempt remains recorded as not started.

The checkpoint sections below preserve their original support policies and
outcomes. Statements there about mandatory 0.8.1 acceptance, a pending floor
decision or then-next implementation work are historical; the current
authorized policy and remaining gates are stated above.

### Exact platform selector and native eye forwarding

One shared contract now validates the exact native
`platform_index(platforms=(("mosaic",), None))` selector: no inputs or effects,
one strong scalar INT32 result and ordinary compiler-domain result 1. SOURCE
admits its conditional only from that successfully validated same-scope producer
with matching two-branch metadata and signatures. Both branch bodies remain
validated. This is not general CPU/GPU/TPU dispatch; no host-backend guess,
source substitution or unselected-branch bypass is used.

The initial three-owner selection completed on all three installed stacks with
**104 unique cases: 103 PASS / one FAIL / zero ERROR or SKIP**. The shared failure
was the SOURCE NaN-diagonal test's public selector witness: its unselected
Mosaic branch called `jnp.eye`, whose existing wrapper substituted a custom
carrier outside the profile. The original native diagonal constructs its eye
before dispatch. These failures remain in `platform-index-v2-{minimum,secondary,primary}/`.

The correction forwards SOURCE `eye` calls to the captured original with exact
arguments before any normalization; the legacy signature and body remain
unchanged. No eye carrier or new branch primitive is admitted. Fresh native and
activated traces match, including nested branch parameters, result dtypes,
shapes and weakness. Tests retain both original NaN witnesses and check native
x64 defaults, explicit FP64 canonicalization with x64 disabled, a rectangular
offset, and exact argument/error forwarding without the private carrier.

The same **110 unique cases pass on PRIMARY, SECONDARY and MINIMUM**, with
**zero failures/errors/skips** and held sources. All 104 earlier identities are
retained and passing; the scope adds two new eye controls and four existing
generated Eye cases. It comprises 17 platform-owner, 52 SOURCE-profile,
37 composite-reuse and four generated Eye cases. Raw XML/log hashes and current
source bytes have independent review. Receipts and raw logs are in
`platform-index-v3-{minimum,secondary,primary}/` under
`/tmp/jax2onnx-restart-checkpoint-20261021/`.

Actual typing passes on **559 source files**; **17 RNG factories** are an
inventory count. Pinned Black/Ruff and all four policies pass; the latter
receipts/logs are in `platform-index-style-policy-v3/`.

The fresh ordinary PRIMARY full passes **11,685 unique cases**, with **zero
failures/errors/skips**, in **638.753850 seconds**. All **11,668** identities from
the preceding BF16 full and all **110** focused identities are retained; exactly
**17** additions belong to the platform test owner. All **1,097** source/config
pins, including **933 Python files** and the five changed code paths, match the
run inputs. Git state was held. The receipt and raw XML/log are in
`platform-index-full-v3-primary/`; XML SHA256 is
`f7f68d4036a853e82574cfaa1bbd49cc40706f93531466b6bac2784b694472e8`
and log SHA256 is
`084724da95dfc46ed001532170184cd8fd375d5da4264d7849430eacb07bf4c3`.

The same **110 unique focused identities** additionally pass on actual extracted
**ONNX Runtime 1.24.1**, with **zero failures/errors/skips**, in **16.862991
seconds**. Source, Git, driver and extracted runtime files were held; this uses
the PRIMARY JAX stack with the declared-minimum ORT. Its receipt and raw XML/log
are in `platform-index-ort124-v3/`; XML SHA256 is
`a8b7ef00066bd15317f519d8d04b380af83fed8bae683447f46f32a6c8522786`
and log SHA256 is
`d797ff7101bc85b4c736c55a029450972225348b9c59d732279cbea889695c86`.
Phases are **NOT_INSTRUMENTED**. This exact ten-path checkpoint is committed at
**`b7eb923a93f642a82002636f8b7bf5999f6713f7`**, parent `3d9ba486`. Strict MkDocs
passes in **6.45 seconds**, with **269 rendered relative-link targets across nine
docs** passing. All seven normal hooks pass. At commit verification, all **933
Python inputs** and qualified committed bytes matched the full run, and the
worktree was clean. Commit records are in `platform-index-commit-v3/`; the hook
log SHA256 is
`a61522d4ad50ef0f1ee13bc24967738551ac149ccaa17e0d6baa2e76283060ae`.
The current contraction and full-map work is recorded above; complete minimum
and B–F acceptance still precede the public transition.

### BF16 CPU sample transport qualification

The closed BF16 full-run pytest model tree was moved from RAM to a verified
durable archive. Original paths remain in raw reports;
`/tmp/jax2onnx-restart-checkpoint-20261021/bf16-full-artifact-relocation-v1.json`
records the mapping. Durable receipts, XML/logs and input pins remain intact,
and the archived models retain their original bytes.


Checkpoint **`b1e1e5f0579f64c083e985435fbe510ef8802f56`** is committed across
seven qualified paths. Its PRIMARY full **11,627 PASS** and later locked
SECONDARY full **11,620 PASS / two known SKIP** have zero failures/errors. The same
**929 Python files** hold across both full runs; all seven normal hooks pass,
committed bytes match qualification and the worktree is clean at verification.
The SECONDARY skips retain their original conditions and do not establish
acceptance after the skipped assertions. Phase counts are **NOT_INSTRUMENTED**.

The later shared sample validator selects BF16 transport from actual BF16 feeds
or physical BF16 artifact outputs before one original execution. Genuine inputs
keep their dtype through same-width storage views; actual CPU output dtype,
shape and byte count authorize an owned BF16 byte copy. Empty outputs require no
pointer read. Ordinary `Session.run`, model bytes, references, comparisons,
evidence format and failure categories are unchanged. No kernel emulation, numeric
cast, reference-driven allocation or retry is introduced.

The preceding `copy_tensors` dynamic proof remains **FAILED** on both ORT
**1.24.1 and 1.31.0**: ten positive runs per runtime returned actual tensors but
readback failed because no data transfer implementation was registered. The
separate bounded owned-byte proof passes on actual ORT **1.24.1 and 1.31.0**, each with **14 sole original runs,
12 nonempty reads, zero empty reads and two preserved Add load failures**.
Its twenty decoded outputs per runtime agree exactly across runtimes and with
independent raw references. These are finite transport measurements.

Applied production qualification passes the same **275 unique cases on each of
PRIMARY, SECONDARY and MINIMUM**, including all **41 new BF16 cases**, with
zero failures/errors/skips and held sources. Actual extracted **ORT 1.24.1**
passes those same **41 new cases**, with source/driver/runtime files held.
This includes exact and tolerant criteria, signed-zero/subnormal/NaN bits,
noncontiguous inputs, scalar/empty/mixed outputs, value-dependent output sizes,
wrong-reference numerical failures and pre-read metadata guards. The original
explicit BF16 arange samples now match on all three stacks; the old constant
omitted-step sample matches on MINIMUM, and legacy dynamic BF16 arange at
opset 23 matches on all three. Modern omitted-step Add still fails runtime
loading; opset 27 remains a runtime-opset failure, and data-dependent SOURCE
arange remains unfinished native tracing work. No SOURCE admission is added.

The first typing run failed on one redundant import-ignore comment; its log
remains retained. The applied comment-only correction preserves the whole
module AST. Actual typing now passes on **557 source files**; pinned Black/Ruff
and all four policies pass. The **17 RNG factories** are an inventory count.
Focused tests precede that correction. The fresh PRIMARY ordinary full binds
the final bytes and passes **11,668 unique cases**, **zero failures/errors/skips**,
in **672.969802 seconds** (ROOT `79263` / `a633f2`, exit0). All **11,627 prior
identities** and exactly **41 new BF16 cases** pass, including every focused
275-case identity. All **1,094 source/input files** hold; phases are
**NOT_INSTRUMENTED**. Final production/test SHA256 prefixes are `35267aed` /
`6403a243`.
This exact seven-path checkpoint is committed at
**`3d9ba486913ce5f9ab75ffcbadb67c9eed0cdda8`**, parent `b1e1e5f0`. All seven
normal hooks pass. At commit verification, committed/worktree bytes and all
**930 Python inputs** from the full run matched qualification, and the worktree
was clean. Strict MkDocs passes in **6.44 seconds**, with **266
rendered relative-link targets across nine docs** passing. Commit records are
in `bf16-runtime-commit-v1/`; its actual hook log has SHA256
`26092f5532bc52ff47d26908fa9d01a7749f89f96f8994045a2fb08ced939ce0`.
Historical NumPy-interface failures and failed numerical reports retain their
original artifacts and epochs.

Durable records under `/tmp/jax2onnx-restart-checkpoint-20261021/` include
`sequence-conv-commit-v1/receipt.json`, `sequence-conv-full-v1-secondary/`,
`bf16-ortvalue-dynamic-v1/{current-v1,floor-v1}/`,
`bf16-ortvalue-dynamic-v2/{current-v1,floor-v1}/`,
`bf16-runtime-adapter-v1-{primary,secondary,minimum}/`,
`bf16-runtime-ort124-v1-primary/` and `bf16-runtime-full-v1-primary/`. The records
retain their receipts and raw logs or tensor payloads; none replaces an earlier
failed epoch. The authoritative new full receipt has SHA256
`cb80e1dc98960eb6a8a3e221292837847c92f3df06a845941cf578fa12230ce1`.

### Restart recovery and native fixture qualification

A machine restart removed the shared-memory evidence and stopped the prior
SECONDARY run before a recoverable terminal result. The preceding PRIMARY
**11,616-case pass** had been observed and independently reviewed; its raw
receipt, XML and payloads are no longer available. It remains a historical
observation and does not qualify the current code. The current regression
record uses fresh durable runs; no result is inferred for the lost SECONDARY
execution.

The renewed Python 3.11 SECONDARY full completed **11,611 unique cases**:
**11,605 PASS / one FAIL / five SKIP / zero ERROR**, with source held. These
ordinary runs have no instrumented phase totals. The failure is the FP16
decoder normalization benchmark; the five skips retain their individual native
or optional-integration conditions. The log, XML and receipt are retained in
`/tmp/jax2onnx-restart-checkpoint-20261021/secondary/`.

That environment still used ONNX **1.22.0** and ONNX Runtime **1.29.0**, while
the unchanged lock specifies ONNX **1.23.2** and ONNX Runtime **1.31.0**, both
compatible with Python 3.11. A diagnostic preserves the original FP16 weights,
inputs, native references and `rtol=atol=0.002`. Native eager and JIT references
match exactly. With 1.29, `auto` fails under full runtime optimization and passes
with optimization disabled; `prefer_native` fails under both settings. Replaying
the same saved artifacts and inputs on 1.31 passes **all eight runtime
configurations**, with maximum absolute error **0.001953125**. This locates the measured discrepancy
in that runtime target. The original failed full and failing 1.29 comparisons
remain failed history; no benchmark code, model, reference, dtype or tolerance was
changed. The failed first diagnostic, which stopped at a typed-key serializer
before references or exports, is retained separately.

The SECONDARY environment has now been synchronized to the existing lock.
Dependency and lock checks pass; JAX/JAXLIB **0.10.2**, Flax **0.12.8** and
NumPy **2.4.6** are unchanged. Fresh SECONDARY execution after
that synchronization now passes the same **234-case** NNX/benchmark selection,
including all seven benchmark tests, with zero failures, errors or skips. The
production converter and repository script contents at checkpoint `2b4882b1`
were unchanged from the durable failed SECONDARY full; those corrections were test
fixtures. The later Conv forwarding correction is recorded separately below.
This focused result preceded the completed locked SECONDARY full recorded above;
it does not relabel the failed earlier full.
The diagnostic and same-artifact replay are in `decoder-diagnostic-v2/`, and
the fresh affected receipt, XML and log are in `nnx-focused-v2-secondary/`.

A separate native-only NNX probe captures the unchanged seed-zero Linear rows
0/3 under both caller-x64 settings, with two fresh instances per setting on
each of the three stacks. Independent audit checks all **324 archived arrays**.
The minimum JAX/JAXLIB **0.8.1** / Flax **0.12.1** kernel differs from the modern
kernel in **29 of 8,192 FP32 values**, by at most **2.9802322387695312e-8**.
The existing bias and feeds match, and each stack's repeated, eager and JIT
observations match exactly. This is native-constructor evidence, not ONNX or
numerical acceptance; its durable JSON and NPZ files are in `nnx/`.

The reviewed shared test fixture recognizes that exact minimum package pair,
original recipe and measured CPU settings while retaining the fixed modern
expectation. Original parameters, source-to-ONNX payload checks and rational
FP64 bounds remain unchanged. The fixture correction is applied locally.
The first minimum affected run remains **FAILED**: **225 PASS / nine FAIL /
zero ERROR or SKIP**, **234 unique cases**. Its fixed-hash guards all advance;
the nine failures expose older placement-argument and variable-hook APIs. The
subsequent fixture-only API correction preserves native/SOURCE behavior and
modern argument coverage. Fresh minimum execution now passes **all 234 cases**
in the four NNX owner modules and the benchmark module, including every prior
failure and all seven benchmark tests, with zero errors or skips. Its durable
record is `nnx-focused-v2-minimum/`; the failed predecessor remains in
`nnx-focused-minimum/`. A later formatter-only correction preserves the Python
syntax tree and has separate verification.

Fresh minimum core validation also passes **1,652 cases**, zero failures,
errors or skips, across nine whole modules: JIT, loops, predicates, typed values,
framework compatibility, typed trace carriers, precision diagnostics, Gather and
probe evidence. The durable receipt, XML and log are in `typed-focused-minimum/`.
Independent XML review confirms **1,886 distinct passing minimum cases** across
these two disjoint selections and the same 234 NNX/benchmark identities before
and after the API correction. Phase counts are **NOT_INSTRUMENTED**. The same
**234 NNX/benchmark cases pass on all three installed stacks**, each with zero
failures, errors or skips.

At the earlier committed `2b4882b1` checkpoint, the ordinary JAX/JAXLIB
**0.12.0** PRIMARY full passed **11,622 unique cases**, with **zero failures,
errors or skips**, source and Git state held
(ROOT `40730` / `3263ba`, exit0). The durable runner records **797.65 seconds**.
Phase counts are **NOT_INSTRUMENTED**; no phase totals are inferred. Its receipt,
XML and log are retained in `primary/` under the durable restart directory.
Independent review confirms retention of all **11,611** identities from the
durable SECONDARY full, plus six new NNX controls and five existing PRIMARY-only
native-conditioned identities. The focused PRIMARY and minimum-core identities
are included and passing. No identity comparison is claimed against the lost
older PRIMARY raw collection.

Checkpoint **`2b4882b1`** is committed with exactly **22 qualified paths**,
verified committed contents and a clean worktree at commit verification. All
**seven normal commit hooks pass**, as does the **strict documentation build**.
Its quality checks pass typing across **557 source files**, four policy checks,
Ruff and formatting on **20 touched Python files**. One pinned-Black formatting
correction has a separately verified unchanged syntax tree. These completed
commit gates qualify that finite checkpoint; subsequent changes retain
their own validation and commit obligations. The complete minimum suite, a fresh
locked SECONDARY full, full B–F acceptance and the public transition remain open.

The historical minimum full below remains **FAILED and unaccepted**. The fresh
focused failures recorded next identify current obligations; repaired-case
counts are not subtracted from an earlier full to invent a remaining count.
Public LEGACY behavior and the declared support floors remain unchanged.

### Fresh minimum baselines and sequence fixture repair

After the committed checkpoint, the unchanged three-owner minimum selection
completes **285 unique cases: 243 PASS / 42 FAIL / zero ERROR or SKIP**. The
failures comprise 16 sequence, 17 constant-normalization and nine interop cases;
both former TypePromotionError-lookup failures now pass. A separate unchanged
16-owner selection completes **1,401 unique cases: 1,372 PASS / 29 FAIL / zero
ERROR or SKIP**. Sources and Git state are held in both runs; phase counts are
**NOT_INSTRUMENTED**. These are separate selections, not a full-minimum result
or a summed acceptance denominator. Durable receipts, XML and logs are in
`remaining-native-baseline-minimum/` and
`other-retained-failures-baseline-minimum/` under the restart directory.

The paired normalization diagnostic invokes eight exact original recipes on
MINIMUM and PRIMARY, retaining their original inputs, parameters, options,
graph checks and assertions. All eight serialized models and feeds are
byte-identical across stacks. Their target outputs are also byte-identical
under original-default, ALL and DISABLED optimization. Native model state,
inputs, eager-before/after bytes and caller settings remain held. Five constant
recipes have nonzero native eager/JIT results on minimum and zero on primary;
the unchanged target is exact zero on both. The three interop recipes have
different eager and JIT results on both stacks and retain the minimum numerical
failures under their original bounds. Changing the native reference to JIT,
substituting an algorithm or widening a tolerance is not acceptance. The paired
captures are in `normalization-diagnostic/{minimum-v1,primary-v1}/`. The subsequent
three-recipe statistics capture completes on both stacks: native grouped inputs
and squares are bit-identical to target, the first observed differences on
MINIMUM are in mean or second moment, and the copied models retain their original
final bytes;
its original minimum failures remain failed, with no causal, impossibility or
acceptance claim.

The sequence fixture correction retains the exact original omitted/explicit
arange calls and contexts. On the old native constant-only route, checks bind
closed constants and their output order to pinned payload bytes and actual
serialized output producers. The seven malformed-form controls now expose
native iota/mul independently of incidental arange constant folding, validate
the clean form and preserve every original corruption/rejection. The first
fixture run remains **FAILED: 115 PASS / 10 FAIL**, with zero errors or skips;
it stopped at an incorrect diagnostic-callback count. After that fixture
correction, the whole module passes the same **125 unique cases on MINIMUM,
SECONDARY and PRIMARY**, each with zero failures, errors or skips and held
sources; phases are **NOT_INSTRUMENTED**. These are focused test results, not a
new full-suite qualification. At that fixture epoch, the old BF16 runtime
interface failure and raising diagnostic remained retained, with SOURCE
admission and numerical acceptance **UNVERIFIED**. The later BF16 sample
boundary qualifies new executions separately; earlier failed sequence epochs
are not relabelled.

The exact subnormal replays on MINIMUM and the locked SECONDARY produce
**48 actual FAILED numerical reports** and **48 raising acceptance-guard checks**,
with zero diagnostic errors. They retain the original bitwise criteria and failed
status; passing diagnostics supply no numerical acceptance.

The LEGACY Conv shim now replays the original caller's exact arguments at its
three existing fallback sites, including the original input before any batch
flattening. Omitted `out_sharding` stays omitted; explicit None, keyword or
positional hints and native errors retain their original behavior. SOURCE
continues to call the native implementation directly. Five new controls cover
the three fallback routes and native result bytes/error parity under both
policies. Legacy `inputs=` remains an independent binding gap, while SOURCE
`x=` retains native rejection; the full call-form obligation remains separate.

The combined sequence, ordinary-matrix, matrix-admission, signed-Min/Max and
Conv-geometry selection now passes the same **412 unique cases on each of
MINIMUM, SECONDARY and PRIMARY**, with zero failures, errors or skips and held
sources. It retains 407 previous identities and adds exactly five reviewed Conv
controls. All 25 failed baseline identities belonging to these five owners now
pass; this is a scoped identity comparison, not a whole-minimum remaining count.
Phase counts are **NOT_INSTRUMENTED**. Current typing passes **557 source files**;
Ruff and formatting pass on the **five changed Python paths**; all **four policy
checks pass**. The 17 RNG factories are an inventory count, not an executed-test
count.

The current ordinary PRIMARY full passes **11,627 unique cases**, with **zero
failures, errors or skips**, source and Git state held, in **694.655 seconds**
(ROOT `44606` / `6b6c4d`, exit0). Phase counts are **NOT_INSTRUMENTED**. Its
receipt, raw XML and log are retained in `sequence-conv-full-v1-primary/` under
the durable restart directory. The strict documentation build passes (ROOT
`37473` / `4dc6de`, exit0, **8.08 seconds**), and **218 relative links across
nine docs pass**. All seven normal hooks subsequently passed and the exact
seven paths were committed at **`b1e1e5f0`**, with qualified bytes and a clean
worktree verified. Its locked SECONDARY full subsequently completed **11,620
PASS / two known SKIP**, zero failures/errors, on the same 929 Python files;
its durable record is `sequence-conv-full-v1-secondary/`.

Other retained minimum obligations include metadata and native-scope schemas,
platform-index lowering, attention numerics and native APIs unavailable in the
oldest packages. Each needs its own source-preserving qualification. No floor
change is authorized while the user's decision is pending: **Python >=3.11 /
JAX >=0.8.1 / Flax >=0.12.1** remains binding. The complete minimum suite,
existing-functionality/fixed-source B–F acceptance remain open. Strict docs,
normal hooks and qualified-byte verification are required for a checkpoint
commit; public behavior remains **LEGACY**.

### Earlier wave-2 and declared-minimum baseline

The wave-2 JAX/JAXLIB **0.12.0** PRIMARY full run passes **11,518 cases /
34,554 phases**, **zero failures, errors or skips**, in **801.272 seconds**
(ROOT `80151` / `723e37`, exit0). ROOT and independent review verify each
collected identity's setup/call/teardown exactly once, matching controller/four
worker collections, normal plugin autoload, four workers and `loadscope`, with
source and inputs held. The original **11,322 / 11,427 / 11,482** identity
sequences remain in order, including all 44 first-run failures; 36 new Erfc
controls follow the first 0.12 collection. A preliminary pure postcheck failed
because it compared stripped and unstripped Git newlines; the exact saved
HEAD/status were subsequently verified. That failed postcheck is retained;
the native full run was not repeated.

The fresh wave-2 first gates separately pass on all three installed native
stacks, with **four positive / 26 negative synthetic applicability controls**.
Those are metadata and synthetic-control results, not numerical proof. The
Python 3.11 SECONDARY affected-20-file selection passes **1,328 cases / 3,984
phases**, **zero failures, errors or skips**, in **348.692 seconds** (ROOT
`18462` / `11c651`, exit0). ROOT and independent review verify each collected
identity's three phases exactly once, the retained ordered affected identities,
matching XML and held source/inputs. Its actual collection was ROOT `92364` /
`53fbe2`, exit0.
This is an affected selection on JAX/JAXLIB 0.10.2 and Flax 0.12.8, not a
complete SECONDARY suite or declared-minimum qualification.

The first JAX 0.12 PRIMARY full run remains closed and **FAILED**:
**11,438 PASS / 44 call failures / zero errors or skips**, **11,482 cases /
34,446 phases** (ROOT `1450` / `d7b9ac`, exit1), with every identity and phase
verified and held source/inputs. Its raw `schedule_complete=false` is an
acceptance-success flag, not evidence of omitted execution. The 36 Erfc
guard/category failures and eight diagnostic-tuple assertions retain their
original source epoch and failed status.

The reviewed five-file correction is applied locally. Erfc admits only absent
or singleton-`None` default accuracy; every non-None request rejects before
either policy accesses values or IR. Its expression, coefficients and local
low-precision casts are unchanged. Two exact JAX 0.12.0 / ORT 1.31.0 tuples
record observed subnormal deviations. All **24** replay reports (**eight
matrix / 16 signed Min/Max**) remain numerically **FAILED**, with original
words/hashes, bitwise criteria and actual raising gates. Passing diagnostic
controls do not relabel those reports. The earlier repair selection passed
**117 cases on PRIMARY and 117 on SECONDARY**, zero failures/errors/skips
(ROOT `73657` / `7c499d`, exit0); those ordinary focused runs have **no
instrumented phase-count proof**.

Wave-2 quality passes **all eight checks**, holding **953 Python / 164 other
inputs**, **1,117 total**, including **582 production Python files**, under
source epoch `dbfc367467f4ef929f3fd23803f13fd2cf54acb1d533f2c9455bd70c659fc18d`.
Typing covers **557 source files**; **17 RNG factories** is an inventory count,
not 17 executed tests. Earlier 94/94 components, 66/64 context, 560-case SOURCE
and metadata gates retain their separate source epochs. Earlier 67-PASS/one-FAIL
and 64-PASS/one-FAIL component executions remain failed history.

The distinct native Erfc v2 capture records **28 cases**, **eight genuine older
public-keyword absences**, and **16 available within-stack default/None
comparisons** with identical raw **StableHLO** and exact eager/compiled output
bytes on its introduced five-value probe. Raw optimized-HLO differences remain
preserved. This is neither cross-version bitwise equivalence nor SOURCE/ONNX/ORT
or universal numerical proof. The failed v1 collector is retained separately.
Actual native Sign stage captures and the bounded **30-file / six-ELF** offline
inventory preserve their own source epoch. Loaded/executed Sign instruction
ownership, relocation-resolved loaded RX bytes and the actual executing worker's
MXCSR/operand state remain open; original Sign numerical reports remain **FAILED**.

The v7 declared-minimum JAX **0.8.1** / Flax **0.12.1** full run remains
**FAILED and unaccepted**: **223 failures / 97 errors / one skip**, **11,419
cases**, pytest exit1, with source/inputs held. ROOT observed all six original
OS processes absent. Tool handle `31003` was lost; the OS record does not invent
an unavailable outer tool exit. The historical v7 PRIMARY **11,427-case /
34,281-phase pass used JAX 0.11.2**, distinct from both current 0.12 epochs.

Declared floors remain **JAX >=0.8.1 / Flax >=0.12.1**, preserving **Python
3.11**. The user reports roughly half of downloads use Python 3.11; this is a
planning constraint, not independently measured telemetry here. A **JAX-only
0.10.2 floor** is a proposal awaiting explicit user decision. **Flax 0.12.8 is
the actually qualified secondary pairing, not a required companion floor**.
Flax 0.12.1 declares `jax>=0.8.1` without an upper bound; retaining that oldest
Flax alongside a proposed JAX 0.10.2 floor needs separate actual qualification.

Before any generalized-contraction packet, the current minimum-repair
obligations remain binding: central old/new PRNG, native aval/equation-context,
JIT configuration and older scan ownership, meaningful corruption/restoration
controls and actual minimum qualification. The exact original obligation is
retained in the [work plan](../precision_implementation_work_plan.md#current-binding-minimum-version-obligation).
No public selector, legacy fallback or numerical-policy waiver is authorized.
The original **1,852-row / 3,704-caller** SOURCE baseline remains unchanged;
the polynomial adapter adds no accepted row. Modern PRIMARY full and affected
SECONDARY qualification and their independent actual reviews are complete at
that wave-2 source epoch. The then-pending local checkpoint hooks and strict
documentation gates are resolved by `2b4882b1`, as recorded above. That finite
checkpoint does not qualify the failed minimum or complete the existing-functionality,
fixed-source, full B–F or public-transition targets.

### Subsequent minimum-stack repairs

Before the restart, the native TypedNdArray adapter preserved owned payloads
and effective weakness across the installed stacks. The initial five-module
run passed **1,420 cases** on MINIMUM; each modern stack recorded **1,417 PASS /
three fixture-preparation failures**. All 37 original minimum typed failures
passed. The corrected typed/alias selection later passed **144 cases per
stack**, and serializer controls passed **36 per stack**. Original failures and
negative criteria were retained; these were focused results.

The Gather and diagnostic correction passed **195 cases per stack**, including
the 97 original Gather setup-error identities. The native-only captures and
trace recognition supplied no exporter, runtime or numerical acceptance. A
subsequent whole-oracle descriptor refresh passed **93 cases per stack**, with
all 14 failures from the preceding **11,602 PASS / 14 FAIL** PRIMARY full
retained. That selection comprised two whole modules plus two exact cases.
Its copied "three whole modules" description was a metadata error; actual child
arguments determine its scope.

The successor PRIMARY full's **11,616 PASS** and the focused results above
were observed before the restart; their shared-memory raw evidence is now
unavailable. Ordinary focused and full runs had no instrumented phase counts.
Earlier quality passes belong to those source epochs. The current fixture
changes require their own affected, quality and full checks before a local
checkpoint. No historical failed numerical report is relabeled.

### Historical evidence references

The following references retain their original source epochs. Shared-memory
paths lost during the restart are historical references, not currently available
raw evidence. Durable restart records are identified above.

| Evidence | SHA256 |
| --- | --- |
| Wave-2 JAX 0.12 full raw receipt: `/dev/shm/jax2onnx-cnn-scientific-runtime-v1/checkpoint-full-primary-jax012-compat-v2/run/root-receipt.json` | `50edccc8d45141120436b05ae4cde387c11e4e5361ee0d6a0bda0a6d8f7ebc89` |
| ROOT wave-2 full actual terminal: `/tmp/jax2onnx-jax012-root-integration-v2/root-full-primary-actual-terminal-v1.json` | `c65e7c634869dd3abc69b3278431bc75c6e9c2f0bff8b3ddeb838b12109d154c` |
| Independent wave-2 full actual audit: `/tmp/jax2onnx-astra-jax012-wave2-primary-full-actual-audit-v1.json` | `68a96c45c19e8afbfaa6fee1c8c86629ae63518f3b5da69173fc9835a8dccba7` |
| Wave-2 first gates/control actual receipt: `/tmp/jax2onnx-jax012-root-integration-v2/root-first-gates-and-controls-actual-v1.json` | `ea9d84c6867fec6b95d05d5f547cb44ad283e61ee0531e773460523cb342ac3c` |
| Independent wave-2 first gates and collection audit: `/tmp/jax2onnx-astra-jax012-wave2-first-gates-collection-audit-v1.json` | `d5050d3a3b6e9d84bdfc65034b6bd926d0ddf0e5c24e5e59fbd64e53f79e41b1` |
| SECONDARY affected20 collection only: `/dev/shm/jax2onnx-cnn-scientific-runtime-v1/checkpoint-affected20-secondary-jax012-compat-v2/collect/root-receipt.json` | `b2f6009c13066a5336ccfd0b59ac96ebd7fa51ae293a4b3eaa4f20f3035b615f` |
| SECONDARY affected20 actual raw receipt: `/dev/shm/jax2onnx-cnn-scientific-runtime-v1/checkpoint-affected20-secondary-jax012-compat-v2/run/root-receipt.json` | `22ef4be3ad177bafe79f6b698f4a8335b3e6ea25e37949e9b01e19bbebe75eaa` |
| ROOT SECONDARY affected20 actual terminal: `/tmp/jax2onnx-jax012-root-integration-v2/root-affected20-secondary-actual-terminal-v1.json` | `cb382bf6022fd4b46590ed7e44b1a5172bbcbf2dd7d01757f8233a3a74e4865e` |
| Independent SECONDARY affected20 actual audit: `/tmp/jax2onnx-astra-jax012-wave2-secondary-affected20-actual-audit-v1.json` | `9b8d7508d7167075caf075ed310c483baabc948b5a9bc3a270ad7576c507a9c6` |
| First JAX 0.12 failed full raw receipt: `/dev/shm/jax2onnx-cnn-scientific-runtime-v1/checkpoint-full-primary-jax012-compat-v1/run/root-receipt.json` | `b8009270de1bea966ca152fe42f6675eaf7ccaef6b4c3ffc83df617e94be3fb1` |
| Applied exact five-file repair: `/tmp/jax2onnx-jax012-root-integration-v1/root-erfc-tuples-application-v1.json` | `43a8eb8afb7ae161458df71946d4e8210b822891e5a548acafdde454b01223bf` |
| Independent 24 FAILED diagnostic witness audit: `/tmp/jax2onnx-astra-jax012-subnormal-witness-audit-v1.json` | `7499a7a680de8a3d401751cdce216390398df9778e72184b25911649ebe49440` |
| ROOT 117/117 focused terminal; phases not instrumented: `/tmp/jax2onnx-jax012-root-integration-v1/root-erfc-focused-terminal-v1.json` | `a956b91683c36c5c3fbc32d7b99033954db3fdf485239a960ad18616f39c3d25` |
| Wave-2 eight-check quality: `/tmp/jax2onnx-jax012-root-integration-v2/root-production-quality-v1.json` | `19b7d98f4e2cdd46b26c05b33828258f93cb64bacc17c9beb23c6360566088d3` |
| Held source/input pins: `/tmp/jax2onnx-jax012-root-integration-v2/source-pins-candidate-v2.json` | `dbfc367467f4ef929f3fd23803f13fd2cf54acb1d533f2c9455bd70c659fc18d` |
| Independent native Erfc v2 audit: `/tmp/jax2onnx-astra-erfc-native-schema-actual-audit-v2.json` | `f3b1265bfa5229c134555fe3bef263c6ea1ec63375666486a25fa2e20eb5b910` |
| Independent native Sign stage-capture audit: `/tmp/jax2onnx-astra-sign-primary-stage-capture-actual-audit-v1.json` | `01d7dc149725350ff0add3e119f3f18c6af87cf7ce93cd60221bdc8302d93adb` |
| Independent Sign offline-inventory audit: `/tmp/jax2onnx-astra-sign-primary-offline-actual-audit-v1.json` | `22edbcc918d546a6887bb690f7656c7e6ab3b45815a5b1535ce9de4aafd1ca5c` |
| Failed v7 minimum raw receipt: `/dev/shm/jax2onnx-cnn-scientific-runtime-v1/checkpoint-full-minimum-jax081-compat-v7/run/root-receipt.json` | `f6a92ab81ad80a009fc927fa852000346e4649d2b2547da45fe8e69d6b11e78f` |
| ROOT minimum OS-terminal limitation: `/tmp/jax2onnx-jax081-compat-root-integration-v7/root-full-minimum-actual-os-terminal-v1.json` | `936b59c9d3c1d3b1d97412229d187fe7d7c2db48fb95daed168b77f16555505b` |
| Historical JAX 0.11.2 v7 full receipt: `/dev/shm/jax2onnx-cnn-scientific-runtime-v1/checkpoint-full-primary-jax081-compat-v7/run/root-receipt.json` | `c284daa8a13c44c73c211c899bbb08f4d14b956ca0e31c4b5525440e966e8f30` |

The earlier findings below retain their original checkpoint scopes. Their
then-next-packet wording is historical; the current qualification obligations
are stated above.

## Canonical native capture checkpoint

Status: **COMMITTED LOCAL CODE CHECKPOINT `98ace5f9`; ACTUAL AFFECTED
DUAL-STACK, FULL PRIMARY, QUALITY AND ALL SEVEN NORMAL CODE HOOKS PASS**.
This packet repairs
exact source/state capture for selected native callables. It adds no SOURCE
admission, target execution, numerical guarantee or public capability.
Public behavior remains **LEGACY** and full B–F acceptance remains open.

The adapter owns the canonical `logistic_p`, `reduce_window_sum_p` and
`reduce_window_max_p` primitives and the `maximum`, `minimum` and `logical_xor`
ufuncs, including their selected `bind`/`reduce` aliases and existing callable
routes. It retains public/defining identities, class and instance state,
Python code/defaults/closures/attributes, native JIT configuration and the
Python cache-miss callback behind the bounded native cache interface. Typed
scalar records distinguish Boolean/integer values and retain complete floating
words. Unknown state and changed bindings remain explicit capture gaps;
unowned key, annotation, descriptor and metaclass callbacks do not run merely
to describe rejected state. Actual native outcomes and arrays are saved before
a changed-state guard rejects further preparation.

The earlier v4 affected executions remain **FAILED** on both stacks:
**808 PASS / ten FAIL out of 818 cases**, with 2,454 phases each. Eight failures
were fixture contamination: restoring an inherited Primitive attribute with
`setattr` left an instance field behind. One control confused an unknown actual
callable with a recursive metadata class token. The retained nonleading-symbol
squeeze test and five original replay cells also exposed an unstable code hash.
No complete PRIMARY runtime was run for that failed epoch.

ROOT's independent stdlib witness on CPython 3.12.9 and 3.11.12 showed that
retaining a nested function changes `marshal.dumps(outer.__code__)` reference
bookkeeping while the outer code object and every constant identity remain
unchanged. Unmarshalling yields equal code objects. The accepted v6 correction
replaces the three code-hash sites with one exact structural fingerprint of
the 18 public `CodeType` constructor fields, recursively typed constants and
complete floating/complex words. Integer encoding also handles values beyond
Python's decimal-string limit. It does not discard code, reset state or warm
models to hide changes; private interpreter slots and universal semantic
equivalence are outside this structural fingerprint claim.

The fixture now restores exact instance-dictionary membership and checks all
six canonical owners. Actual unsupported class callables are rejected at the
target boundary, while recursive type metadata and the existing exact JAX
scalar constructor adapter remain supported. All 166 original capture test
IDs remain, with 41 added controls covering code fields, nested constants,
reference stability, unsupported constants and target ownership.

Actual corrected collections retain all prior IDs in order: **859 focused
cases = 652 historical affected cases + 207 capture controls** per stack, and
**11,322 full PRIMARY cases = 11,115 historical passing cases + 207 additions**.
All 818 IDs from the failed focused epoch and all 11,281 IDs from its full
collection are also retained in order, with 41 additions. The corrected
focused executions pass **859 cases / 2,577 unique phases on each stack**,
zero failures, errors or skips, with identical ordered IDs and actual process
exits zero. All ten previously failed IDs pass. Independent review checked
saved native outcomes, retained native-contract/scientific controls and the
original Issue52 input words and criteria; those existing finite comparisons
do not establish new provider-wide guarantees.

The complete PRIMARY execution passes **11,322 cases / 33,966 unique phases**,
zero failures, errors or skips. All four worker collections and controller
views agree with the frozen collection; all five processes exit zero.
Independent review verifies every setup/call/teardown phase exactly once,
the XML results and both historical ID sets above (ROOT session `77044`,
terminal `a79dea`, exit0).

Fresh quality passes all eight checks, including production mypy on **555
source files**. A strict documentation build at the code checkpoint also passes. All
**945 Python / 164 other inputs** hold through quality, focused and full
executions and the documentation check.

Code checkpoint **`98ace5f9a68d02481434ac1551058d14a417f231`**, parent
`4a9260c0bf4664c7fb56be284c99888cba7ed08f`, contains the exact three qualified
paths: the compatibility module, raw replay helper and native capture tests.
All seven actual normal code hooks pass; all **945 Python / 164 other inputs**
retain their qualified contents through commit, and the worktree is clean at
verification (ROOT session `39778`, terminal `2c6ad0`, exit0).

The exact original replay selects 17 unchanged rows and both x64 caller
scenarios, **34 cells per stack**, from the unchanged **1,852-row /
3,704-caller-cell** plan. Each stack records **30 native-valid cells, four
native failures and zero capture gaps**. Ten artifacts pass schema/lowering
checks and their original callbacks; 20 cells remain SOURCE profile exclusions.
All ten artifacts are byte-identical to the earlier epoch. Independent
comparison of 224 available old/new saved-array roles per stack preserves
dtype, shape, storage order and array payload bytes, without treating NPZ
container timestamps as data.

The four native failures are the x64-on original `reduce` rows 0/1 and
`reduce_window` rows 1/2: FP32 operands meet native FP64 weak initial values.
Eager execution, fresh JIT and both trace modes raise `TypeError`. The first
three were already native failures in v1; row 2's same eager failure was saved
before the old code-hash gap, and its remaining modes are now recorded. No
input, initial value or original source was cast or rewritten. The old v1
outcomes—ten artifact passes, 16 exclusions, three native failures and five
capture gaps—remain immutable history. Both replay epochs preserve opset 23,
all original options/callbacks and `criteria=None`; they execute no ONNX
Runtime and leave runtime/numerics **UNVERIFIED**.

The advertised **JAX 0.8.1 minimum now has an actual import defect**. An isolated
native-only probe of compat `606f0824` raises `ModuleNotFoundError` for the newer
PRNG module path. The v6 capture correction leaves that path unchanged. This
later evidence supersedes the earlier minimum-version UNVERIFIED statements
retained below; it does not constitute exporter, Flax, model or ORT qualification.
The next separate repair owns old/new PRNG paths, native aval fields, recorded
equation-context activation/restoration, all 21/22 JIT configuration fields and
the older scan parameters, with minimum-stack and both current-stack evidence.
The supported minimum and full functionality remain requirements.

| Evidence | SHA256 |
| --- | --- |
| Retained failed v4 dual execution: `/tmp/jax2onnx-native-capture-root-integration-v1/root-dual-focused-failed-actual-terminals-v1.json` | `a97c44c0a4145adb01bd8b60c8bd44207c7c758bf0651856bfa3a1b247e5eb07` |
| Actual marshal reference witness: `/tmp/jax2onnx-marshal-reference-bookkeeping-root-v1.json` | `3376fe7fa1a636c9361d541e8f2a17361d9751f75b6ca8f1920da10313b595f4` |
| Exact v6 static acceptance: `/tmp/jax2onnx-astra-canonical-native-capture-static-review-v6.json` | `3eb7452a13583fedff2d01a27c5aa27ae14c97ec7631cdcace35815c8d4b27cc` |
| Actual corrected quality: `/tmp/jax2onnx-native-capture-root-integration-v2/root-production-quality-v1.json` | `59e914815ea0383c273e8da6ce6454cffaecd86119300e8a17e7e938b9aab294` |
| Actual corrected collections: `/tmp/jax2onnx-native-capture-root-integration-v2/root-collections-actual-terminals-v1.json` | `c5b909c87ae01bbf3e9caf155dad126001fe17fb5fe53675452181511d7af151` |
| Actual corrected dual terminals: `/tmp/jax2onnx-native-capture-root-integration-v2/root-dual-focused-actual-terminals-v1.json` | `1f1f1de635fffd7eadd1cef75eaecfc0e23e871a2ee9b5947750583835a3d12a` |
| Independent dual review: `/tmp/jax2onnx-astra-native-capture-focused-audit-v2.json` | `46845b52efd69f8d0cca5dbe9d80746eb397d536e39179dae02c5044e45aef34` |
| Actual full PRIMARY terminal: `/tmp/jax2onnx-native-capture-root-integration-v2/root-full-primary-actual-terminal-v1.json` | `ce6c2b23f55fe82950e711a93085db9e09613e69496df6fa381dd22f3479ce8d` |
| Actual strict current docs: `/tmp/jax2onnx-native-capture-root-integration-v2/root-code-docs-strict-v1.json` | `178376c8d608f1634550c9b911290e44e43c67be89e228107cfe9c91592f60b0` |
| Independent final code-commit readiness: `/tmp/jax2onnx-astra-native-capture-full-readiness-bind-v2.json` | `eecd183b8d63a5204dd32a4968bdccb978ed5043e609b3591c18412fa05090a3` |
| Actual code commit and normal hooks: `/tmp/jax2onnx-native-capture-root-integration-v2/root-code-commit-v1.json` | `56aeb3effcaccbda39e0d651dd96afd949dfb3635fdc0f36434026fa96270317` |
| Original 17-row actual terminals: `/tmp/jax2onnx-native-capture-root-integration-v2/root-original17-actual-terminals-v1.json` | `8d65792cc3229fd8bb5c95d5273717a9511e00a49c57b886ec8dbf9a60c3e33a` |
| Independent original replay/array audit: `/tmp/jax2onnx-astra-native-capture-original17-dual-audit-v2.json` | `63cbe5c2dfb577f54ca06bd5f24a11eaa71163ebe94658f3c2110595455c32c1` |
| Actual JAX 0.8.1 native diagnostic: `/tmp/jax2onnx-jax081-native-runtime-v1/native-probe-v1.stdout.json` | `dbb5d1acce3e8023bf854b1bad8f97e5041604e58a93f7e39bb317bbf99242ce` |
| Separate minimum-version repair plan: `/tmp/jax2onnx-astra-jax081-compatibility-handoff-v1.md` | `138532a211a3ca22fa78965e0eb8f38086640f69a8beb387b19fd10015d79a26` |

The complete earlier findings body below remains byte-identical. Its failed
ordering, FP16/subnormal, Erfc, Sign and raw callback diagnostics retain their
own criteria and statuses. This bounded capture checkpoint does not complete
the full raw map, fixed-source workflows, numerical acceptance or B–F migration.

## Shared native result contract checkpoint

Status: **COMMITTED LOCAL CODE CHECKPOINT `00d0a43c`; ACTUAL FOCUSED DUAL-STACK,
FULL PRIMARY, QUALITY AND ALL SEVEN NORMAL CODE HOOKS PASS**. This shared
validation refactor applies to existing owners. It adds no SOURCE admission, target numerical evidence,
provider guarantee or public capability. Public behavior remains **LEGACY**;
full B–F acceptance remains open.

The compatibility boundary freezes the installed canonical evaluator and native
result convention for scan, while, scatter, scatter-add, scatter-mul and remat
before plugin binding. It uses the original avals and parameters inside the real
recorded `JaxprEqnContext.manager`, with exact prior-context restoration in
`finally`. It rejects unsupported sharding, manual-axis, memory, layout and
quantization roles on inputs, declared outputs and inferred outputs. Only exact
ordinary `ShapedArray` values with the supported default metadata pass.

The shared converter check requires canonical primitive identity, the owner's
explicit parameter fields, exact result count/dtype/shape/Boolean weakness,
native symbolic-scope identity and both declared and inferred empty effects.
Existing owner-specific capability, parameter, child-body, port, capture and
recursion checks remain in place before inference. Native abstract evaluators
can rely on body declarations; matching their results alone proves neither
child semantics, target lowering nor numerical agreement. Recorded equation
context does not imply replay of unrelated x64, promotion or matmul settings.

Qualification covers the installed JAX 0.10.2 and 0.11.2 stacks. This helper's
compatibility with the advertised JAX 0.8.1 minimum remains **UNVERIFIED**; no
minimum-version defect or fallback is established by those two-stack results.

Actual collections retain all prior IDs in order: **627 focused cases per
stack = 544 retained + 83 additions**, and **11,115 full PRIMARY cases = 11,032
retained + 83 additions**. The focused executions pass **627 cases / 1,881
unique phases on each stack**, zero failures, errors or skips, with actual
process exits zero and source/input pins held. These controls cover the six
selected families, canonical-evaluator/result-flag tampering, identity,
parameters, result metadata, symbolic scope, effects and native context
restoration.

The separate complete PRIMARY execution passes **11,115 cases / 33,345 unique
phases**, zero failures, errors or skips. It retains all **11,032 prior IDs** in
order with exactly **83 additions**, exits zero and holds all **944 Python /
164 other inputs** (ROOT session `63894`, terminal `8809f7`). This regression
result leaves the separate failed numerical diagnostics unchanged.

Actual quality v3 passes all eight checks, including production mypy on **555
source files**; its 17 RNG factory entries are a report inventory, not tests.
All **944 Python / 164 other inputs** hold through that quality execution.
Earlier failed epochs remain **FAILED**: v1 stopped at the new test file's
missing path header, and v2 reached typing and failed six context-`None`
diagnostics. The accepted header-only successor preserves the complete test
AST (`1f48cd68`), and the accepted explicit-`None` context guard yields compat
`039e6b5b`. These narrow corrections precede v3 and the actual focused runs;
they do not erase either failed epoch.

Code checkpoint **`00d0a43c2743c099996bf0b8fba7ef8136b41046`**, parent
`ac842e15cd93efabb922c5eb5c06897918e07b70`, contains the exact six qualified
paths. All seven actual normal code hooks pass; all **944 Python / 164 other
inputs** retain their qualified contents through commit, and the worktree is
clean at verification (ROOT session `84922`, terminal `fe7fde`, exit0). The
accepted header and explicit-`None` corrections above are included in those
committed bytes.

| Evidence | SHA256 |
| --- | --- |
| Independent six-file static review: `/tmp/jax2onnx-astra-source-native-contract-sharing-static-review-v1.md` | `3f03c5b46f015e5e7b563ecf523ce73f5fdbc74bd5b2064327169707d7d82674` |
| Retained failed quality v1: `/tmp/jax2onnx-native-contract-root-integration-v1/root-quality-failed-v1.json` | `ff5534da54923105352a3cf272b17854753d694a7faeecd8991f9f26b416a4d2` |
| Retained failed quality v2: `/tmp/jax2onnx-native-contract-root-integration-v2/root-quality-failed-v1.json` | `8580a3d03309b426b6b4677d4ddbfbfc1796bc93b06ec3e82d99b1291f46544d` |
| Actual quality v3: `/tmp/jax2onnx-native-contract-root-integration-v3/root-production-quality-v1.json` | `150ef74fcb9902920e43e67595d6b501f30ece7a51caf1519e0e343be5330ff2` |
| Actual quality/collection terminals: `/tmp/jax2onnx-native-contract-root-integration-v3/root-actual-quality-and-collections-v1.json` | `e98197da180c3a4b7562f10e0ffc850114d14788f34c907fa1aa3c130e183104` |
| Actual focused dual terminals: `/tmp/jax2onnx-native-contract-root-integration-v3/root-dual-focused-actual-terminals-v1.json` | `07f3afb38622701692941749a53c58a09675dd4813c1bcbf33c4d90a8ed9de86` |
| Full PRIMARY actual terminal: `/tmp/jax2onnx-native-contract-root-integration-v3/root-full-primary-actual-terminal-v1.json` | `0129c6139f8c1dcc0703e25dcca331d4d7a705447b53afc489b9d943c3971150` |
| Code commit actual terminal: `/tmp/jax2onnx-native-contract-root-integration-v3/root-code-commit-00d0a43c-actual-terminal-v1.json` | `a8d1fe13d795d35ddfb1affdf90d84a2757b4e50491edb017d1a00b454deda4e` |

The complete earlier findings body below remains unchanged. Its complex Add,
CNN/scientific samples and partial raw map retain their original scope. Ordering
kernels remain **FAILED** with SOURCE admission closed; native-bitwise Min/Max
subnormal evidence remains **FAILED**. Original raw callbacks and `criteria=None`
are unchanged, provider-wide numerical guarantees remain **UNVERIFIED**, and the
full 1,852-row / 3,704-caller map and B–F migration remain open.

## Complex Add and partial ordering checkpoint

Status: **COMMITTED LOCAL CODE CHECKPOINT `18f3fcd7`; PRIVATE FINITE COMPLEX
ADD QUALIFICATION; ORDERING DIAGNOSTIC FAILED; FULL MIGRATION OPEN**. Public behavior remains **LEGACY**. The following
scopes are separate source epochs; none completes all existing functionality,
fixed-source workflows or B–F.

### Native complex Add

The private route admits the actual parameter-free native Add with two operands
and one result of the same complex width with static shapes at opsets 23/26.
Complex64 uses
FLOAT real/imaginary pairs and complex128 uses DOUBLE pairs. Both inputs must
match their exact declared physical dtype/shape before lowering; complete result
geometry comes from the native output declaration. Ordinary real trailing-two
tensors keep their real meaning. Shared representation and strict core output
validation remain unchanged.

Finite controls cover scalar/vector/logical trailing-two/empty/broadcast shapes,
all native weakness combinations, captured values, named/JIT scopes, real
siblings and repeated outputs. Complex64 is checked with caller x64 both off
and on. Complex128 is checked with x64 already effective and values that expose
FP32 downcast; the explicit complex128/x64-off conflict and original stored
capture controls remain. Mixed widths, real/complex mixing, other complex
arithmetic and symbolic forms remain implementation gaps, not impossibility
claims.

The 148 added cases pass on each supported stack. The complete focused
selection passes **2,082 cases / 6,246 unique phases per stack**, zero failures,
errors or skips, with identical ordered IDs and source/input pins held.
Native eager/fresh-JIT references, exact artifacts, both CPU optimization modes,
arrays and optimized graphs are retained before acceptance assertions. The
finite exact dyadic predicate is declared before targets; it does not establish
provider-wide numerical behavior.

The complete PRIMARY suite passes **11,032 cases / 33,096 unique phases**,
zero failures, errors or skips. All **10,884 historical IDs** are retained in
order with exactly **148 additions**; the actual process exits zero with source/
input pins held (ROOT session `67393`, terminal `cfdaad`). These test results
retain the separate FAILED numerical diagnostics described below and in history.

Code checkpoint **`18f3fcd70459896905c067662eaccdcdf76e4bbf`**, parent
`6f547bb64c3483d567ee01c8c19c9d68998ec0b9`, contains exactly the four reviewed
and qualified paths. All seven actual normal hooks pass; all **941 Python /
164 other inputs** retain their qualified contents through commit, and the
worktree is clean at verification (ROOT session `20452`, terminal `103c23`,
exit0). Fresh pinned Black/Ruff on those four Python paths, four whole-source
policies and production mypy on **554 sources** also pass. These are finite
checkpoint gates; the broader migration and numerical limits remain open.

All six original Add caller cells per stack now pass product artifact/schema
checks and their unchanged raw graph callbacks. These product-stage reports
still have runtime/numerics **UNVERIFIED**; original `criteria=None` remains.
The supplemental finite comparisons do not fabricate raw criteria, and the
initial B000 complex exclusions remain immutable history.

### Partial initial raw map

The b000–b011 aggregate binds 24 actual cohorts from the preceding `118ddd8b`
code epoch: **332 original rows / 664 caller cells per stack**, leaving **3,040
caller cells unobserved per stack**. All original options, data, callbacks,
criteria, and source/stage failures remain. This initial map performed no ORT
execution; its runtime/numerical statuses remain **UNVERIFIED**. It is neither
the full 1,852-row / 3,704-caller map nor coverage of the later complex packet.
The six-cell Add correction above is a separate source epoch.

### Ordering kernels: actual failures retained

The standalone [ordering kernel probe](ordering_kernel_probe.py) compares native
eager/fresh JIT with plain ArgMin/ArgMax/TopK kernels, using unchanged predeclared
input words, exact index/component predicates and index/value association.
Its tracked executable body matches the actual reviewed v2 script; relocation
does not rerun it. These are supplemental kernels, with no SOURCE export,
original nine-row replay or raw criterion replacement.

Each stack records **32 FAILED / four passed of 36 grouped kernels** at original
opset23 and CPU optimization off/all. The four passes are sampled BOOL ArgMin/
ArgMax with exact BOOL-to-INT64 encoding. All native eager/JIT observations agree
within their stack; target off/all outputs and target outputs across stacks
agree. The failures retain these distinct causes:

| Scope | Actual boundary |
| --- | --- |
| FP16 ArgMin/ArgMax | Later-position/multiple NaNs produce indices different from native references. Sampled finite ties, zeros, infinities and subnormals agree. |
| FP32/FP64 ArgMin/ArgMax | NaN differences plus subnormal index differences; this is numerical behavior, not a missing CPU kernel. |
| FP16/FP32/FP64 Approx min/max through TopK | Tied selections at k=1, NaNs and some signed-zero/exceptional selections disagree; FP32/FP64 also differ on subnormals. Unique ordinary finite rows agree. |
| BF16 ArgMin/ArgMax | Schema passes, while both CPU modes report missing ArgMin/ArgMax(13) implementations. The probe provides typed BF16 transport; session creation fails before that transport is reached. |
| BF16 TopK at opset23 | Schema rejects the input type and CPU reports INVALID_GRAPH; this is separate from the Arg kernel gap. |

Approx top-k does not supply one portable stable tie order: the installed
PRIMARY documentation explicitly states that tied selection is unstable, and
both installed comparators ignore index arguments. Observed k=1 index3 versus
TopK index0 therefore does not prove a universal native last-index policy.
Native k=2/full results differ between stacks only on NaN-containing rows.
The frozen stronger exact-index comparisons remain **FAILED**; no new policy
for permitted tied subsets is introduced.

A separate native association failure remains visible. FP32/FP64 Approx k=1
returns positive zero with index4 on the recorded subnormal-1 row, whose original
element4 has nonzero positive-subnormal word1. Both native roles fail exact
original-byte association in all 16 corresponding checks per stack; every target
association check passes. Exact agreement with those native words and the
unchanged association predicate cannot both pass for that witness. This is a
specific backend/sample conflict, not an ONNX impossibility claim. No FTZ,
input/reference/predicate rewrite, floating cast or opset upgrade closes it.

Ordering SOURCE admission stays closed. Arg NaN propagation, multiple-NaN
selection and subnormal behavior need owning semantics; Approx needs separate
option/comparator/selection ownership. Nondefault aggregation/recall/override
options, other widths/shapes, x64-off behavior and broader compositions remain
unqualified. Exact integer results cannot be accepted by widening a floating
tolerance. The original nine rows retain opset23, options, data and
`criteria=None`.

Saved ordering summaries are under
`/dev/shm/jax2onnx-cnn-scientific-runtime-v1/ordering-bare-kernel-dual-opset23-v2/{primary,secondary}/summary.json`.

| Evidence | SHA256 |
| --- | --- |
| Code commit actual terminal: `/tmp/jax2onnx-complex-add-root-integration-v1/root-code-commit-18f3fcd7-actual-terminal-v1.json` | `10ede5cc115f2f22b25f0b344c226b99dfd9f711bea6a68dda84ee64264e5712` |
| Fresh production quality: `/tmp/jax2onnx-complex-add-root-integration-v1/root-production-quality-v1.json` | `dde9f18dd81260f8e6a479470cbbc589d1f07c8e3b763b5cc9a629a024cab726` |
| Actual focused dual terminals: `/tmp/jax2onnx-complex-add-root-integration-v1/root-dual-focused-v2-actual-terminals.json` | `38bae3d90c9ff061e60b318d8216dc9e58221909691cb97b562c86ab7bf63f35` |
| Fresh quality and original-six product terminals | `e2336e70c6325337383efefc003254e8bf52955bd116660b67bcf153e3fadd68` |
| Full PRIMARY actual terminal: `/tmp/jax2onnx-complex-add-root-integration-v1/root-full-primary-actual-terminal-v1.json` | `fdbbb898bc4cf6a1b3a4e501767dbd292676610e6da52dadb2e2ed8a0403bbe0` |
| Initial b000–b011 aggregate: `/dev/shm/jax2onnx-cnn-scientific-runtime-v1/raw-first-barrier-map-b000-b011-summary-v2.json` | `4ab3f7191fe12444b8af88c72dd3e0e97736729ac8d77d080116c93a2c96e428` |
| Actual ordering v2 PRIMARY summary | `133551d070d0f5a00ea10c2fd59a530d5cb1679928c95aa16487dee47f63a11a` |
| Actual ordering v2 SECONDARY summary | `21c5643496f82ede7a0ddd4d206ff29b0fdca4c2081d0304b2717118cbd248bf` |
| Executed ordering v2 script | `804cac4d0a930bb311b6b268a864b0bea751bfb22e8522379ce6859ccc8728fc` |
| Independent ordering review: `/tmp/jax2onnx-astra-source-ordering-kernel-review-v2.md` | `dca334223132fe4de98559cd162efe3dc8c6a191169554272c4431720c53f1dd` |

The earlier CNN/scientific/raw-lane checkpoint below retains its complete body
and then-current statuses byte for byte. Its nonzero CNN predicate remains
finite, with provider-wide guarantees **UNVERIFIED**; original raw callback
failures/missing criteria and native-bitwise Min/Max subnormal **FAILED** evidence
are unchanged. The full B–F migration, fixed-source coverage and public
transition remain open.

## CNN pooling, scientific FP64 and raw replay checkpoint

Status: **COMMITTED LOCAL CODE CHECKPOINT `118ddd8b`; INDEPENDENT FINITE
ACCEPTANCE, 1,798 CASES PER STACK, FULL PRIMARY 10,884 AND ALL SEVEN NORMAL
CODE HOOKS PASS**.
The current source has 940 Python and 164 other inputs (1,104 total), including
579 package Python paths. Public behavior remains **LEGACY**. This finite
checkpoint does not complete B–F or run the full 1,852-row / 3,704-caller map.

### Scoped implementation and numerical evidence

The raw replay lane preserves original recipes, options, values, state, caller
settings and independent native roles. It separates capture, tracing, product
export and original callback outcomes. Its 49 maintained controls cover the
reviewed lane and known gaps; they do not execute the full denominator or
establish model-wide numerical acceptance. Original callback failures and raw
`criteria=None` remain unchanged.

The CNN path retains the complete original FP32 model, all eight parameter
arrays, nonzero feeds, static/symbolic batch contracts and named ownership.
Native disjoint window sums remain separate from division by four. Signed-zero
ReLU selection is checked as a complete owned composite, including its exact
selector casts; unsupported floating data casts still fail. Weak FLOAT pool
sums convert to DOUBLE before Div only where the actual native trace requires
it. Existing Conv tests retain their stricter no-Cast contract.

The original nonzero CNN samples use the fixed a priori exact-input forward
roundoff predicate, with exact dyadic centers and outward-rounded error bounds.
It is neither a fitted tolerance nor a theorem about an opaque provider's
algorithms. The provider-wide theorem stays **UNVERIFIED**. Separate finite
receipts do not replace original raw callbacks, missing raw criteria or stored
export reports' **UNVERIFIED** execution/numerical statuses.

Selected scientific FP64 elementary, scatter, remat and nested scan contracts
now include opset 21. The original Issue52 helper, NPZ, input bytes and
`rtol=1e-5, atol=1e-6` remain unchanged. Its boundary check permits rank-only
annotations solely for five proven unused inner Loop constant-state outputs;
initial inputs, body IO, mutable carry, outer and stacked outputs remain strict.
Native and both CPU target arrays/optimized graphs are saved before acceptance
assertions. All metadata corruptions must fail the structural helper.

The Min/Max subnormal regressions have two obligations: exact nonzero source
operand selection, and the unchanged native-bitwise criterion. The latter
remains **FAILED**, with runtime execution **VALIDATED** and a raising acceptance
guard. Passing the paired diagnostic means that it detected this unmet target
criterion. It does not establish native bitwise agreement, an ONNX impossibility
or a new public precision mode.

### Actual qualification and retained failures

| Gate | Actual result or current limit |
| --- | --- |
| Initial PRIMARY diagnostic37 | 1,788 PASS / 10 FAIL of 1,798; 5,394 unique phases, zero errors/skips, source/input pins held. Five failures concerned Issue52/MinMax assertions and five concerned CNN/pool test helpers. This complete failed epoch remains failed evidence. |
| Finite modules in that failed epoch | Raw replay: 49 cases / 147 phases PASS; nonzero CNN: 46 / 138 PASS; Linen forwarding: 10 / 30 PASS. These are subscopes of the failed epoch, not another aggregate acceptance run. |
| Corrected scientific selection, each stack | 141 cases / 423 unique phases PASS; zero failed/error/skip, actual exit 0 and source/input pins held. Original Issue52 runs both CPU modes against fresh eager/JIT references. Native-bitwise subnormal samples remain FAILED inside passing diagnostics. |
| Current quality v2 | Fresh pinned Black/Ruff on 26 changed Python paths plus four whole policies PASS. Prior actual mypy on 554 sources and 17 RNG factory report entries carries on byte-identical 579 package files. The 17 entries are not 17 executed tests. This is not full-suite or numerical acceptance. |
| Corrected 37-module selection, each stack | **1,798 cases / 5,394 unique phases PASS**, zero failures/errors/skips; exact same ordered IDs, actual exit 0 and source/input pins held on PRIMARY and SECONDARY. |
| Full PRIMARY | **10,884 cases / 32,652 unique phases PASS**, zero failures/errors/skips. All four workers and controller-seen IDs match actual collection; every process exits zero and source/input pins hold. The 10,596 prior IDs remain in order after the exact four CNN anchor mappings, with 288 additions. |
| Independent final acceptance | The post-full-run readiness review accepts this finite checkpoint. It independently checks saved CNN envelopes/paired allowances, Issue52 outputs, failed native-bitwise subnormal evidence, source bindings and completed-run artifacts. Its then-pending hook/commit gates are resolved by the later ROOT receipt below. |
| Normal code hooks and commit | All seven actual hooks **PASS**. Code commit **`118ddd8be1623c5bb890c072aabd7e651525b9c9`**, parent `97929998`, contains exactly 28 qualified paths. All 940 Python / 164 other input contents remain byte-identical through commit; the worktree is clean at verification. ROOT session **88673**, terminal **1ca828**, exits **0**. |
| Code-checkpoint strict MkDocs | Actual code-checkpoint build **PASS** in 10.00 seconds; ROOT session **36353**, terminal **f86eb9**, exit **0**. |

The preceding Linear checkpoint's strict documentation build did pass and its
eight-document update is committed at **`97929998`**, after code commit
`b2a866bb`. Any then-pending wording in the retained checkpoint below records
that earlier point in time. Historical mentions of 17 RNG checks refer to
factory report entries, not 17 executed test cases.

| Current evidence | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-cnn-scientific-root-integration-v1/root-diagnostic37-primary-v1-actual-terminal.json` | `9aff7e7f29833b68218594d029519dd34afbf659df40d60f1ced7071185395c2` |
| `/tmp/jax2onnx-cnn-scientific-root-integration-v1/root-science-contract-dual-v2-terminal.json` | `6965f6633cc8bb459fca05b3e6a08de8d447fbb48fa42e44e7dfabf824f8e85e` |
| `/tmp/jax2onnx-cnn-scientific-root-integration-v1/root-production-quality-v2.json` | `dea725487b024150ebd3eb81c2c45aecf94f649ee28d897a5592596116723507` |
| `/tmp/jax2onnx-cnn-scientific-root-integration-v1/root-corrected37-dual-v2-actual-terminals.json` | `d9490e1b98ca381a617b944156459ac6efca00ca0e9188b7edf1aba8040283db` |
| `/tmp/jax2onnx-cnn-scientific-root-integration-v1/root-full-primary-v1-actual-terminal.json` | `f877a9bc9398a2c576ccd25780e93f08679fbe0a742e82f8083fed454cd52981` |
| `/tmp/jax2onnx-astra-cnn-scientific-checkpoint-readiness-v1.md` | `83537768e998fa2ec3c47fb7eb9138c085d800cc842528b809764252f1ea945d` |
| `/tmp/jax2onnx-cnn-scientific-root-integration-v1/root-code-docs-strict-v1-actual-terminal.json` | `490c4276b9dd0c90c6a9923f96ec56d9994bc9400a65a711ce705a940098b4d6` |
| `/tmp/jax2onnx-cnn-scientific-root-integration-v1/root-code-commit-118ddd8b-v1.json` | `22d7ac87831d10e37f1f9fed17c7807db6444db98f0abbd20e73c8475f365933` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-docs-commit-97929998.json` | `fd4c37419711146bc265f435215c565dc4959fd309d5ecb14c7c86d06b78014f` |

Next run and review the full raw inventory map. The complete map has not run; bounded lane controls are not that
execution. Preserve all original unresolved criteria, callback failures,
optional-environment obligations, fixed-source workflows and genuine target
limits. Broader normalization/models, low precision and full B–F remain open.
The following earlier sections retain their historical bodies and then-current
status; this section supersedes only the exact gates explicitly named above.


## Explicit FP64 Linear preparation checkpoint

Status: **COMMITTED LOCAL CODE CHECKPOINT `b2a866bb`; ACCEPTED DUAL936
ON UNCHANGED PRODUCTION, FRESH DUAL145 AND FULL PRIMARY V4 PASS; ALL SEVEN
NORMAL CODE HOOKS PASS**. This finite explicit Linear-preparation checkpoint
does not complete general scientific FP64 support or change public **LEGACY**
behavior. Strict documentation build and the separate documentation commit
remain pending.

### Implemented behavior and numerical boundary

The private preparation accepts an already constructed exact `nnx.Linear` and
returns an ordinary NNX clone configured for explicitly requested FP64
computation. It retains the supplied model source, complete captured weights,
options, graph aliases, state, Variable metadata and promotion/dot callables.
Omitted storage conversion retains the actual parameter values and declaration;
explicit FP64 storage converts complete cloned kernel/bias values. Existing FP64
payloads never pass through FP32. The original object and its native behavior
remain available. Widening stored FP32 values cannot recover precision lost at
creation.

JAX x64 must already be effective before preparation. This helper neither enables
x64 nor rewrites arbitrary computations globally; the caller selects and restores
its own local configuration. Unsupported subclasses, storage conversion with
setter hooks, storage families and conflicting preferred result types retain
explicit boundaries.
Ordinary SOURCE validation remains in force after preparation.

Native `Precision` enum pairs, including HIGH/HIGHEST, are additionally admitted
only when both matrix inputs and the result are FP64, with absent or FP64
`preferred_element_type`. The existing shape, weakness, layout and physical
storage checks remain. This does not admit general nondefault matrix precision,
FP16/FP32 HIGH/HIGHEST, precision presets or explicit algorithms.

The maintained original raw Linear rows **0 and 3** retain their **128×64**
weights, original seed **0**, original FP32 B2 input values and an additional B5
feed. The prepared parameterization has **24 cases and 96 actual target returns
per stack**: both raw models, caller-x64 contexts, DEFAULT/HIGH/HIGHEST and
retained/converted storage, using one saved symbolic artifact per case in both
CPU optimization modes. Original eager and JIT references are captured before
SOURCE export.

The separate a priori sample criterion uses exact `Fraction` arithmetic:
`|target-native| <= 2*gamma129*(sum_i(abs(x_i*w_i))+abs(b))`, with
`u=2**-53` and `gamma129=129*u/(1-129*u)`. Its finite normal-range and no-overflow
assumptions are checked from the original operands before target execution.
Shapes, FP64 result types and finite outputs are separate exact checks. The two
runtime modes' observed byte equality is an additional finite sample control,
not a portable bitwise-FP64 guarantee. Custom getter/promotion/dot functions do
not borrow that bound.

The original raw **Gemm callbacks remain FAILED**: SOURCE emits native MatMul
and optional Add. Their metadata and missing numerical criterion
**`criteria=None` remain unchanged**. Stored export reports still record
runtime execution and numerical acceptance **UNVERIFIED**. The separate frozen
finite comparisons establish only their stated sample result.

### Actual qualification and retained failures

| Evidence epoch | Actual result and limit |
| --- | --- |
| Focused v1, each stack | **933 PASS / three FAIL of 936**, zero errors/skips. All 92 added cases pass. Three existing diagnostic assertions require stale default-only wording; these remain failed runs. |
| Rejected regex remedy v1 | Rejected without application or runtime; no passing test epoch. The reviewed v2 aligns exactly three diagnostic expectations in two existing files, preserving production and intended FP16/FP32 rejections. |
| Focused v2, each stack | ROOT-observed actual exit **120 / ENOSPC**. Terminal receipts/XML are empty or invalid; phase/process evidence is missing. Outcomes cannot be reconstructed as passes. |
| Focused v3, each stack | **936 cases / 2,808 phases PASS**, zero failures/errors/skips and all 92 additions pass. Both actual terminals exit zero and every phase is recorded exactly once. Independent acceptance binds the nine-path epoch. All **577 physical production paths** remain identical after the later two test/tool pin updates; this is an accepted production bridge, not a fresh936 rerun. |
| Full PRIMARY v3 | **10,559 PASS / 37 FAIL of 10,596**, zero errors/skips; all **31,788 phase keys** are present exactly once. Actual ROOT process exits **1**. This complete failed execution remains unaccepted. |
| Audited provenance remedy | Applied exactly two hash literals: the accepted DOT module pin and its derived approved counterpart digest. All 11 original dot metadata recipes, lambdas and contracts remain unchanged; all approval predicates and raw callback/numerical rules remain intact. |
| Unrun focused3 harness v1 | Rejected for its list/tuple selector serialization contract before runtime. Corrected v2 receives static acceptance; neither is a failed product-runtime epoch. |
| Fresh corpus3 v4, each stack | **145 cases / 435 phases PASS**, zero failures/errors/skips, covering all three affected modules and **all 37 formerly failed IDs**. Independent review accepts this actual separate post-pin epoch. |
| Full PRIMARY v4 | **10,596 cases / 31,788 phases PASS**, zero failures/errors/skips. Controller and all four workers hold identical ordered IDs; every process exits zero, every phase passes exactly once, and source/Git pins hold. Actual ROOT terminal: session **61508**, chunk **cb0f4f**, exit **0**. |
| Final eleven-path quality | Python 3.11 AST, Black, Ruff and four whole policies pass. Prior actual **552-file mypy plus 17 RNG checks** carries on the identical **577 physical production paths**. All **930 Python / 163 other input files (1,093 total)** hold. |
| Final independent readiness | **436 checks PASS**, accepting this finite eleven-path checkpoint and binding actual full v4, fresh dual145, original dual936 bridge, source/environment/quality and all failed history. |
| Normal code hooks and commit | All seven actual hooks **PASS**: Black, Ruff legacy-alias hook, reflection, mypy, markers, builder and annotations. Actual ROOT session **9612**, terminal **019492**, exits **0**. Code commit **`b2a866bb3e8c579cd2595be28eb318a86438d747`**, parent `454b0c5e90291297072ccb77e5cb67398db647e5`, contains exactly the eleven qualified blobs; all 1,093 input contents hold through hooks/commit and the worktree is clean at verification. |
| Documentation gates | Strict MkDocs and the separate documentation commit remain **PENDING**. |

The failed full v3 first-barrier audit retains **19 direct diagnostics**
(14 DOT source-pin and five dependent Linear module-pin errors) separately from
**18 static dependency attributions** (14 baseline-control failures and four
Linear artifact-stage failures whose tracebacks omit source detail). The later
passing epochs do not rewrite those diagnostic limits or the failed receipts.

The exact committed scope is seven implementation/PoC/test paths, two existing
diagnostic-expectation files and two audited test/tool provenance constants:

- `jax2onnx/_compat/jax.py`
- `jax2onnx/_source_preparation.py`
- `jax2onnx/converter/source_profile.py`
- `jax2onnx/plugins/jax/lax/dot_general.py`
- `internal_docs/precision/poc/fixed_source_preparation.py`
- `tests/extra_tests/converter/test_fixed_source_preparation_poc.py`
- `tests/extra_tests/converter/test_source_fp64_preparation.py`
- `tests/extra_tests/converter/test_source_function_context.py`
- `tests/extra_tests/converter/test_source_nnx_linear_reshape_dispatch.py`
- `scripts/source_precision_probe.py`
- `tests/extra_tests/converter/source_precision_corpus_oracles.py`

| Current evidence record | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-code-commit-b2a866bb.json` | `6538322d034684d215f340ec58377d36144db2ee9027522ea993eefda159abb0` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/fp64-full-v4-final-code-checkpoint-readiness.json` | `853e14ba7ea0386e22860ffb0cbfbd7e07f2a87e3bdaf3c423d457a1383b622f` |
| `/tmp/jax2onnx-fp64-root-integration-v1/code-commit-v1.log` | `f06fa6caf16c173373ba1ec7e12d62ef3e4b8469a4c52ac8b8e4e3299d5c9539` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-full-primary-v4-actual-terminal.json` | `98abf61e23c79b51e02b58285a52889d52e8d49d8fd365f8f7993b785c6e8847` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/fp64-dual-corpus145-v4-actual-review.json` | `4200894037ddfa391483411d1e8fc108c8f8414fe979e8a2a22bd617ce94e011` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-fp64-corpus-pin-quality-pass.json` | `0c43829485ad49ea66e9ff26f9f15099396f9d3a8bef20bd2d68de31de7c1aae` |
| `/tmp/jax2onnx-fp64-root-integration-v1/source-pins-after-corpus-pin-remedy-v1.json` | `e7c2fde43fbaf473a3fb9a2e871589b91d7b696dd75410dc0d69720db8c8e270` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-full-primary-v3-actual-terminal.json` | `6b0e772ffa61ac345261253a4aa79c9fe8aacbb303eabb7f140766d9ddaba712` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/fp64-corpus-pin-remedy-v1-static-review.json` | `2d409bef682736fba117c2808d0d64b73f4b65492b1396b36d16305a52a0a53e` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-corpus-pin-remedy-application-v1.json` | `5de27c02fc302e7aa59a48fa7599cc20b8450b125cab5fda0ec0e2d21d6358e8` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/fp64-dual-focused-v3-actual-review.json` | `da2d900a77d5e1fb4a71351a1f508aa58f490f732e8c8952830c00f667715bbe` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/fp64-corpus-pin-focused3-v2-and-failed-full-v3-review.json` | `4ad749362e28c21b953e49f5dbd5d9d7f6bdd86c1021e6921b9a06cd9946876b` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-focused-primary-v3-actual-terminal.json` | `4956ff80c65c15a22704ac60da1aebc3bcdb4be45dadbec35b94da540a9f5276` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-focused-secondary-v3-actual-terminal.json` | `9e12e44ad5e590b7cc4da5413f7221c7237dc536ae41675c5313f90dfd3875f4` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-fp64-context-quality-pass.json` | `67bfe86a419217782705ec9f11d5f7da5f9b22035b19705c91c6a0653efacabc` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-fp64-focused936-v1-failed.json` | `f5cc7aff475eb4c7d3f698f883ed9178a0f63ec98dcc2f1e1ed0cfdb136ccf4e` |
| `/tmp/jax2onnx-fp64-root-integration-v1/root-fp64-focused-v2-enospc-terminal.json` | `bf07c7acb8298c7d9f62806a49a74b82ca44a0967706059947cb2e06c7995bb4` |

The final quality record carries the original actual mypy/RNG result through
byte-identical production and binds the exact eleven-path/source-pin lineage.
Its then-unrun runtime labels are superseded by the later actual dual145 and
full v4 records. The independent readiness review precedes staging; the later
ROOT commit receipt supplies the actual hook/commit result and verifies every
committed blob. The 17 RNG checks belong to the prior actual quality run, not
to the fresh normal mypy hook.

### Remaining full-scope work

Next qualify the unchanged CNN on its original nonzero input and the next
scientific batch. Raw replay v4 still awaits independent static review and a
future runtime epoch; the FP16 Add/Mul repair has static acceptance but its
native/target and cost gates remain pending. The **22 original compositions /
44 numerically unverified artifacts** still need their explicit comparison
criteria and finite acceptance.

Issue52's opset-21 FP64-input admission gap, general fixed-source preparation,
broader matrices/loops/scientific workloads, BF16 execution/numerics, the FP32
reciprocal observation, eight original failed Sign reports per stack, strict
Erfc full/native failures, failed original Gather callbacks and all 50 optional
obligations remain open. The inventory stays **1,852 raw rows / 3,704 caller
obligations**. Full phases B–F still require one ordinary source-preserving
export across every existing capability and fixed third-party source; no mixed
public precision interface or activation is added.

### Issue52 and opset-21 Min/Max test-contract correction

Status: **WORK IN PROGRESS; TEST/DOCS CORRECTION AWAITS ROOT QUALIFICATION**.
The [actual captures](/tmp/jax2onnx-cnn-scientific-root-integration-v1/root-original-scientific-and-dual-minmax-actual-captures-v1.json)
and [independent decision](/tmp/jax2onnx-astra-issue52-minmax21-test-contract-decision-v1.md)
support a bounded test correction. Issue52's unchanged saved PRIMARY artifact
`339f16a5` has exact DOUBLE output shapes `(5,210,1,1)` and `(2,)`.
All four actual-first comparisons against fresh native eager/JIT references
satisfy the original `rtol=1e-5, atol=1e-6`; maximum absolute difference is
`7.788436562350398e-11`. Its original structural assertion still failed.
These measurements do not establish a SECONDARY Issue52 runtime pass.

The correction permits rank-only annotations solely for unused internal
Loop-body outputs proven to be native constant state, with matching dtype/rank
and intact constant identity chains. Graph IO, mutable state, outer Loop and
stacked results retain exact physical checks, as do initial-state inputs.
Contradictory known dimensions fail. Production postprocessing and arithmetic
are unchanged.

The FP64 Min/Max witnesses check exact nonzero source operand selection
independently of native-bitwise qualification. Fresh native eager/JIT return
signed zeros on both measured CPU stacks; both ORT profiles select the nonzero
operands. JAX documents [backend-dependent subnormal behavior](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#miscellaneous-divergences-from-numpy).
That behavior does not establish a deliberate FTZ requirement in the typed
source graph or an ONNX impossibility. Every unchanged native-bitwise sample
must retain runtime execution **VALIDATED**, numerical acceptance **FAILED**
and a raising acceptance gate, following the [opset-23 precedent](minmax_and_corpus_replay_findings.md#finite-minmax-implementation).
A passing diagnostic confirms the unmet criterion; it does not certify native
bitwise agreement or turn a historical failure into a pass. A caller requiring
those native zero bits still has an unmet target requirement. Deliberate source
conversions, rounding, criteria, public API and precision modes are unchanged.

In `test_source_scientific_fp64.py`, the four historical failed IDs
`test_fp64_minmax21_preserves_zero_polarity_nan_classes_and_subnormals[subnormal-{disabled,all}-{min,max}]`
map to the two proposed
`test_fp64_minmax21_subnormal_source_selection_and_native_failed_diagnostic[{min,max}]`
IDs. Each diagnostic records both CPU profiles and both original native roles
before assertions. Historical failed results remain retained in the linked
captures and initial dual263 receipts; these renamed IDs are a forecast, not a
collected or accepted runtime result.

## Historical checkpoint record

The following pre-FP64 record is retained unchanged, including its then-current
next steps. Later results supersede only the exact gates named above.



Local implementation and review, updated 2026-10-20, beyond the public-behavior
baseline at `fae17822`. All eight saved artifacts across the original six source
selectors now have independently accepted finite CPU samples on each supported
stack. Those artifacts use the frozen `11748451`-derived snapshot. Current MAIN
Conv/Scan integration is committed at `3223d080` after finite independent
acceptance, the complete PRIMARY regression and all seven normal code hooks.
The focused SECONDARY legacy fusion skips remain explicit below. Public behavior stays
**LEGACY**.

The latest saved-sample and implementation outcomes are recorded first. All
following checkpoint sections retain their historical source epochs, counts,
failed results and then-current next steps. A later result supersedes only the
gate and exact artifact/source binding it explicitly identifies.

## Original-six saved samples and native Conv integration

Status: **ALL EIGHT SAVED ARTIFACTS PER STACK HAVE INDEPENDENT FINITE SAMPLE
ACCEPTANCE; THE FINITE CONV/SCAN CODE CHECKPOINT IS COMMITTED AT `3223d080`**.
The accepted samples belong to the unchanged original normalized014/015 models
and their **six source selectors / eight saved artifacts per stack**, including
the two omitted-default exports. They use the immutable `11748451`-derived
**1,087-source-file snapshot**. This is separate from the current MAIN Conv/Scan
code, its regression runs and any future public transition.

### Complete saved-artifact follow-through

Independent review accepts bounded source/parameter/state correspondence and
the exact captured target samples for all eight artifacts on each supported
stack. The original complete models, actual captured weights, state, inputs,
flags and graph callbacks are retained. These are not smaller replacement
models, new MAIN exports, or universal equivalence claims.

| Actual target epoch | Sessions | Captured returns | Scope |
| --- | --- | --- | --- |
| Earlier PRIMARY original014/0 | 2 | 32 | First saved explicit-state artifact; original acceptance retained |
| PRIMARY cell-1 pair | 4 | 96 | Explicit-state and omitted-default artifacts |
| PRIMARY remaining five artifacts | 10 | 256 | All other PRIMARY artifacts under their own frozen criteria |
| PRIMARY total | **16** | **384** | All eight artifacts across six selectors |
| SECONDARY total | **16** | **384** | Its own eight artifacts, source snapshot and native references |
| Both stacks | **32** | **768** | Separate actual sessions and saved arrays; no new trace/export/native evaluation in these target runs |

The same saved artifacts cover the applicable static B5 or symbolic B2/B5
samples, zero/nonzero inputs, T/F/F/T runtime flags, exact uint32 state feedback,
reset and two replays. Every return is checked against four retained native
roles: original eager, original NNX JIT, prepared eager and prepared JIT. Both
CPU optimization modes and both replays return identical target bytes for these
samples; zero-input floating outputs retain exact positive-zero bytes.
The CPU sessions use sequential execution, one intra/inter-op thread and no
custom session configuration or provider options.

Original014 keeps its **reference-first rtol=3e-4 / atol=2e-6**. Original015 uses a
**separately proposed and independently accepted strict rtol=0.002 / atol=0**,
frozen before its target observations. Its historical `atol=None` and source
scenario `criteria=None` remain intact; zero was not reconstructed as an old
harness default. Shapes, dtypes, finiteness and state remain separate exact
checks. Across the accepted samples, maximum absolute error is
**7.450580596923828e-6**, and the largest selected-tolerance fraction is
**0.5151289333544**. These measurements do not define a tolerance for new samples.

The original callback verdicts are unchanged on each stack:

| Saved artifact | Original raw callback |
| --- | --- |
| cell-0 explicit-state | FAILED |
| cell-1 explicit-state | FAILED |
| cell-1 omitted-default | PASSED |
| cell-2 explicit-state | FAILED |
| cell-3 explicit-state | FAILED |
| cell-4 explicit-state | FAILED |
| cell-4 omitted-default | PASSED |
| cell-5 explicit-state | FAILED |

Thus each stack retains **six FAILED / two PASSED** raw callbacks. Existing
export reports still record runtime/numerics **UNVERIFIED**; separate saved
sample receipts carry the new acceptance. No original pytest result, raw
callback, stored report or failed epoch is rewritten.

### Capture and structural limits retained

The PRIMARY continuation still exits **99** at its final raw-eager counter
guard, and its final runtime import-origin record is still missing. All five
selected cells, seven returned exports and 288 native outputs were retained
before that failure. The later offline audit checks the saved computation and
explains why only two raw preparations evaluate while three require live
Boolean preparation. It does not turn that run into an exit-zero result or
reconstruct missing origins. Its independently reviewed summary-label shadowing
defect remains documented; the saved preparation outcomes and callback verdicts
are preserved separately.

SECONDARY completes its own capture with **six cells, eight exports and 320
native outputs**, with zero ORT execution in that capture epoch. Independent
offline review checks 869 saved files and both before/after origin records.
Subsequent structural review covers all eight artifacts; their executable model
fields match the corresponding PRIMARY artifacts, with the recorded JAX-version
report metadata difference. These structural/capture results remain separate
from the later 16-session SECONDARY target-sample acceptance.

### Masked FP64 saved sample

The unchanged `masked_gather_trig_f64` artifact and original inputs now have
independently accepted finite target samples on both installed ORT stacks, with
CPU optimization disabled and enabled. All **four target returns** are bitwise
identical. Eight comparisons / 48 scalar values pass the original
reference-first **rtol=5e-8 / atol=1e-12** against the saved PRIMARY eager and JIT
references; maximum absolute error is **4.440892098500626e-16**. Four existing
sample-API validation reports are accepted separately.

This is separate evidence from the original-six artifacts above. It uses one
PRIMARY-produced artifact and its saved PRIMARY native references. It establishes no fresh SECONDARY native execution or whole-family
acceptance. The original export report's runtime/numeric fields remain
**UNVERIFIED**; separate actual sample reports carry this result. Issue52's
unchanged opset-21 case with three FP64 inputs still has a genuine input-admission
gap. Its proposed engineering repair remains future work.

### Current product implementation and next work

The six native FP32 Conv paths plus the two-path Scan metadata remedy are
committed at **`3223d0800b70d31753db34ffb36ed809654b6f87`**, parent
`c2876fc8`. All **928 Python / 163 other input files (1,091 total)** hold through
the actual full run, all seven normal code hooks and the commit. The eight
committed blobs match the qualified bytes, and the worktree and index are clean
at commit verification. Native Conv admission preserves actual FP32 layouts,
shapes, options and weakness at opset 23; SOURCE NNX/Linen calls forward their
original arguments. Scan retains the declared physical shape while child
Functions lower; the existing later rank-only postprocessing and LEGACY behavior
are preserved. Original metadata and missing numerical criteria are unchanged.

| Current Conv/Scan evidence epoch | Actual result and limit |
| --- | --- |
| Focused v1 PRIMARY | **669 PASS / two FAIL of 671**, zero skips; both failures are real Function-in-Scan Conv consumers receiving prematurely relaxed physical shapes. |
| Focused v1 SECONDARY | **666 PASS / two FAIL / three SKIP of 671**; the skips are existing optional legacy ConvTranspose+BatchNorm runtime-fusion tests. |
| Focused v2 PRIMARY | **675 PASS / 12 FAIL of 687**, zero skips; both former Conv/Scan failures pass. Ten new fixture failures compare inline native bodies with deliberate Function carriers; two insist on an inaccurate diagnostic word. |
| Focused v2 SECONDARY | **672 PASS / 12 FAIL / three SKIP of 687**; the same new fixture failures and unchanged legacy runtime-fusion skips remain. |
| Focused v3 PRIMARY | **687 PASS / 2,061 phases**, zero failures/errors/skips; actual pytest and strict wrapper exit zero. |
| Focused v3 SECONDARY | **684 PASS / three SKIP**, pytest exit zero but **strict wrapper exit 99**. All 586 converter-selection cases pass on each stack; no SOURCE case skips. The three legacy checks remain numerically unexecuted after their skip. |
| Reviewed test-only correction | Applied without production changes. Native references, exact Function context/call wiring and actual exported native bodies are checked independently; all 16 new cases and their feeds/saved bitwise gates are retained. |
| Current quality checks | Normal production typing **551 files** and **17 RNG checks** pass. Final fixture AST/Black/Ruff pass; production typing bytes are unchanged. |
| Complete PRIMARY v3 regression | **10,504 cases / 31,512 phases PASS**, zero failures/errors/skips, actual pytest and strict wrapper exit zero in **1,040.057526669 seconds**. All four workers collect the same 10,504 identities; all **1,091 source/input files** hold. Earlier 10,488/10,504 collections remain collection-only. Independent finite-checkpoint review accepts the exact eight-path code change. All seven normal code hooks pass; committed at `3223d080` with qualified bytes held. |

The corrected counts above follow actual phase/XML evidence. Earlier ROOT
summaries overstated SECONDARY passes by three; their original receipts remain
unchanged. The three skips come from unchanged legacy optional ConvTranspose+
BatchNorm fusion tests (IR, static NNX and symbolic NNX) when that runtime cannot
fuse the path. Numerical assertions after the skip did not execute. This is a
provider-specific evidence gap, not an ONNX impossibility, export-capability loss,
or SOURCE skip. The wrapper exit 99 is retained as a failed strict accounting
gate. Independent review accepts the finite code checkpoint with that limit;
the skipped numerical tails remain unexecuted.

Both earlier failed focused epochs remain retained. All 17 raw Conv rows, all 51 raw
loop rows and the remaining model/precision obligations stay in the inventory.
Continue the unchanged CNN's actual activation/pooling path and its original
Relu/AveragePool/Gemm graph requirements; the next CNN implementation is external
work in progress. Qualify and integrate the separately accepted **seven-path
scientific FP64 candidate** next: its **92 cases are forecast only**, and it
remains **unapplied and unexecuted**. The actual full Linear parameters and
frozen arithmetic criterion must remain its controls.

The full B–F goal remains one ordinary source-preserving export contract across
all existing functionality and fixed third-party sources. Known Sign/strict-Erfc
failures, the 44 numerically unverified compositions, all 50 optional obligations,
low-precision rounding, scientific workloads and general end-to-end acceptance
remain open. The **1,852-row / 3,704-caller** inventory and **318 elementary raw
contracts / 636 caller obligations** are unchanged. Public behavior stays
**LEGACY**; no permanent public selector or temporary subset completes migration.

| Current saved-evidence record | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-conv-root-integration-v1/root-code-commit-3223d080.json` | `a37702b1e8bfa48cb3243f06d8ed71a21e53a20fe813b36299067ff4d229825e` |
| `/tmp/jax2onnx-loop-root-integration-v1/astra-conv-v3-code-checkpoint-readiness.json` | `7ec98b9a8083ba175539cef44e9dc0e312721d21a755b85a8b1bb7a75882d9de` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-scientific-masked-f64-samples-acceptance.json` | `88be00dc05ac365a6683a557c3b89d33e1a50c9d4e5a5a06135c0b734ba9b86b` |
| `/tmp/jax2onnx-conv-root-integration-v1/checkpoint-full-primary-v3/run/root-receipt.json` | `73516b2bcfa1d724478336b217de890c5115ba8c2b91f4d466d0b964483632c6` |
| `/tmp/jax2onnx-conv-root-integration-v1/root-conv-actual-focus-accounting-correction.json` | `c741189f6ff8a12363806e2cc0cf5f561bdcc2fa42f0a15030f3ae1732bfd1e8` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-primary-cell1-two-saved-samples-acceptance.json` | `e08a6e0edc73329dde3de47b80deb6508f0fe6aac4c15c8e065d23546f7ae853` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-primary-remaining-five-saved-samples-acceptance.json` | `42140f0da7153606253e008a39664c009897bd5692605140d8412d7fb5827171` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-secondary-eight-saved-samples-acceptance.json` | `bfff2d72c3b46fe9a2fc0499d2632f5bff0145a6c86eec482bf8c2a2ed9fe20c` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/original015-new-strict-target-proposal-acceptance.json` | `86c4aa4e3744c869909b4019b97b886d5d7c4f33985f037159bb8f49816011d1` |
| `/tmp/jax2onnx-original6-remaining-primary-audit-root-v3/audit.json` | `3e753407cc0960892f380652af6853c459faafa46a2fd95cdd9945131d3c2e97` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-primary-saved-capture-audit-v3-acceptance.json` | `cc4c75b81296fa990337661241124dc8d47a0349bc2441a30a6373cc27ab7f37` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-secondary-saved-capture-offline-acceptance.json` | `7c54b726e5d666d3e4532287d5d47b7b544d86efd03206fa528f315ddc2b9a5c` |
| `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-secondary-eight-artifacts-structural-review.json` | `05d93ce4b3684afad2fed97cc10ad2bab631e6ed6516395991bd5719d91a24f7` |

## Loop admission and first saved-model samples

Status: **COMMITTED LOCAL CODE CHECKPOINT `c033d301`; ACTUAL DUAL-STACK
FOCUSED AND COMPLETE PRIMARY ACCEPTANCE; SEVEN NORMAL CODE HOOKS PASS**.
The commit is **`c033d3013b559ba228e1f73c5f3ffb8783f0f8a8`**, with parent
`003b3e59436a533b917486796067b5d587c781a3`: seven Python files, 1,294 insertions
and 14 deletions, whose committed bytes match the qualified sources. The
worktree is clean at commit verification. The exact seven paths include the native loop compatibility and
admission rules, the narrowly typed zeros carrier, and their existing/new tests.
The frozen execution inputs contain **925 Python / 163 other files (1,088 total)**.
The first saved-model continuation is separate evidence from a
`11748451`-derived immutable snapshot, rather than qualification of the loop code.

| Evidence epoch | Actual result and limit |
| --- | --- |
| Focused loop v1, each stack | **436 PASS / four FAIL of 440**, zero errors/skips; retained failures identify missing ordinary zeros admission and a typed-key diagnostic spelling mismatch. |
| Focused loop v2, each stack | **462 PASS / 15 FAIL of 477**, zero errors/skips; fourteen malformed probes unexpectedly traced native `broadcast_in_dim` instead of the intended zeros carrier, and the complex-negative probe failed before the intended zeros rejection. This failed epoch remains recorded. |
| Focused loop v3, each stack | **477 cases / 1,431 phases PASS**, zero failures/errors/skips, held source/input bytes and independent acceptance. All previous 440 identities are retained. |
| Current complete PRIMARY | **10,395 cases / 31,185 phases PASS**, zero failures/errors/skips, 989.181 seconds. All four workers match the fresh ordered collection, which retains all preceding **10,237** cases plus **158** additions; the controller and four workers exit zero. All 1,088 source/input pins hold. |
| Current code qualification | Independent full-run/readiness review accepts the finite checkpoint. Normal code commit `c033d301` passes all seven actual hooks with no bypass; all 1,088 source/input pins hold through hooks and commit. Strict documentation build and the separate docs commit remain subsequent gates. No push, merge, publication or public precision transition is claimed. |

### Ordinary SOURCE loop admission

Native forward `scan` is admitted at opsets **23/26**, and native `while` at
**18/23/26**, with every actual child operation retaining its own typed/opset
checks. This includes the unchanged original while row 20 at opset 18 and the
original `unroll=4` scan. JAX 0.11 flat-tree partitions and the older explicit
const/carry counts are checked at the compatibility boundary. Native result
weakness is retained separately from carry dtype/shape invariants.

The finite rules preserve real/discrete/static-complex storage, captured values,
zero-trip and empty results, nested JIT/If/Function bodies, and both Boolean
runtime choices through the same artifact. Every child body is validated,
including a zero-trip body. Only the existing strong static FP32/FP64
`jax.numpy.zeros` carrier is newly admitted at opsets 23/26; other creation
wrappers and dtype/opset routes remain excluded. Original metadata, callables,
inputs, opsets, callbacks and missing numerical criteria remain unchanged; the
existing typed-key negative test only aligns its diagnostic regex. Corrected
test fixtures inspect native zero creation without substituting their own
selected-dtype semantics.

These results do not close all **51 raw loop rows**. Symbolic/logical-key loop
transport, unsupported transformed descriptors, reverse scan and remaining
family/model contracts retain their explicit implementation or migration gaps.
All **318 elementary raw contracts / 636 caller obligations**, the complete
**1,852-row / 3,704-caller inventory**, and all **50 optional obligations** remain.
Eight original Sign numerical reports per stack stay **FAILED**, 44 original
composition artifacts stay numerically **UNVERIFIED**, and the strict Erfc
full/native gates and original Gather raw callback failures are unchanged.

### Saved original014/0: bounded structure and target samples

The original first PRIMARY test remains **FAILED** at its unchanged raw graph
callback. Its saved artifact is schema/ports-valid; the callback's direct-root
inspection misses inputs captured by the If branches and the wrapped SuperBlock
path. The earlier stored export report still records runtime/numerics
**UNVERIFIED**, and the original capture-only epoch retains zero ORT calls.

A later independent offline review closes the exact saved artifact's remaining
serialized-expansion fidelity gap. Disjoint producer/dependency proofs account
for all **1,725 stochastic and 157 deterministic Function nodes**, including the
native Threefry/bitcast/Erfc expression, both masks, second Linear and all state
positions. It finds no mismatch for this binding. This bounded structural
acceptance neither changes the raw callback nor establishes universal kernel
equivalence or other model/stack cells.

ROOT then executes this exact saved artifact in **two CPU sessions / 32 calls**,
with runtime optimization disabled and enabled. Both ORT 1.31.0 sessions use
CPUExecutionProvider, sequential execution, one intra/inter-op thread and no
custom session configuration. The same captured B5-zero and B5-nonzero inputs,
T/F/F/T flags, state feedback and four native reference roles
produce **128 comparisons / 19,200 floating values**, with exact count/key state.
The original reference-first criterion remains **rtol=3e-4, atol=2e-6**. Maximum
absolute error is **7.450580596923828e-6**; the worst observed error uses
**0.09051417824110118** of its allowed tolerance. Both modes and repeated replays
return byte-identical target arrays. No new trace, export or native evaluation
occurs in this saved-artifact run. Independent actual-sample review accepts
**only these exact captured samples**.

The other five PRIMARY and all six SECONDARY original cells remain pending
acceptance. The separate PRIMARY capture continuation exits **99** at the final
`run_capture.py:384` counter guard: `raw_eager_evaluations` is **2**, whereas
`factory_limit` is **5**. All **five selected cells**, **seven returned exports**
and **288 native outputs** were retained before this failure, with held saved
source/input pins and **zero ORT constructors/runs**. This is a failed capture
epoch, not a successful run; artifact-body fidelity and target numerics remain
**UNVERIFIED**. Ordinal015 retains its own frozen atol rather than inheriting an
older default. No tolerance, source, input,
callback or model substitution closes these obligations. The saved014 result
does not rewrite its failed raw test or claim whole-model, other-input,
other-stack or public acceptance.

### Next implementation and evidence gates

Continue the remaining original-model cells under their own frozen criteria,
then reconcile each structural/runtime/numerical outcome without replacing the
raw failures. A separate **six-path native FP32 Conv candidate** has independent
source/static acceptance and **93 forecast cases only**. It is **unapplied**:
no collection, runtime or full-package typing qualification is established.
The unchanged CNN and its actual next blocker remain required, rather than a
smaller replacement model. A separate scientific FP64 preparation implementation
is external work in progress; it is unapplied and has no runtime qualification.

The full B–F goal remains one ordinary source-preserving public contract over
all existing functionality and fixed third-party sources. General fixed-source
preparation, optional models, FP16/BF16 rounding, scientific FP64 and full
end-to-end acceptance remain open. No permanent public mode/profile selector or
per-operation LEGACY fallback completes the migration.

| Current evidence | Local record | SHA256 |
| --- | --- | --- |
| Actual loop dual477 v3 | `/tmp/jax2onnx-loop-root-integration-v1/root-loop-actual-dual477-v3-pass.json` | `989172c294756923794dde5b332ee18c031fadc43e8d0ae63b42460c8bad1999` |
| Actual complete PRIMARY v3 | `/tmp/jax2onnx-loop-root-integration-v1/checkpoint-full-primary-v3/root-actual-run-terminal.json` | `1000d9dbb0feb987f6ac82daf5dd64bc14e4fcb55bb984a0537bdc44a69729de` |
| Independent full-run/readiness acceptance | `/tmp/jax2onnx-loop-root-integration-v1/astra-loop-v3-code-checkpoint-readiness.json` | `08fbe696733569b6b0100b57b11380049afc18b84cc87ca61b827873fd9d0a2a` |
| Actual code commit and seven hooks | `/tmp/jax2onnx-loop-root-integration-v1/root-code-commit-c033d301.json` | `7a03765b8a3b8e7855d408a9e16ad1c7196a60eba82af4a0fe54232211de97c2` |
| Remaining PRIMARY failed capture receipt | `/tmp/jax2onnx-original6-remaining-root-v1/results/primary/capture/receipt.json` | `47305391d482b02b694f4f8d1688d8fe884236d5802b852411fa5bfba90512f8` |
| Saved014 ROOT actual terminal | `/tmp/jax2onnx-original6-saved014-root-v1/root-actual-terminal.json` | `dedc273986c03c6ef72461746b465caa3c8396bf7a9aa9dc2522a445c523506e` |
| Independent saved014 structure | `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-original6-first-expanded-source-fidelity-review.json` | `9d73346cc3fc5157446bccbf74d8fa8b25a6fb8e96a1c7c3c5b1fc1f26b3b210` |
| Independent actual saved014 sample acceptance | `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-original6-saved014-samples-acceptance.json` | `ade7a692793598077f681d688a5318a8fb8bb6630888e6e28c00c101cb405f4f` |

## Elementary and reduction checkpoint

Status: **COMMITTED LOCAL CODE CHECKPOINT `11748451`; ACTUAL DUAL-STACK
FOCUSED AND COMPLETE PRIMARY REGRESSION ACCEPTED WITH RETAINED NUMERICAL LIMITS**.
The commit is **`11748451bb0d89f6e0265ea6fec6d982223397ab`**, with parent HEAD `d52a2276`, after code checkpoint `1533d60b`. It changes **86 Python
paths: 63 production and 23 test/support, 4,155 insertions / 178 deletions**.
All committed file bytes match the qualified sources; all **923 Python / 163
other input pins** hold through actual hooks and commit. The worktree is clean
at commit verification. This is local acceptance; no push, merge, publication
or public precision transition is claimed.

The earlier dual183 cohort and its actual artifact/array acceptance remain a
separate epoch. Subsequent reviewed scope, fixture and capture repairs retain
original models and numerical criteria. One enum import adjustment routes through
the compatibility layer and preserves both installed stacks' native semantics.
The complete PRIMARY v4 and dual focused15 v2 results below qualify the final
bytes; earlier failed executions remain unchanged. No complete SECONDARY suite
or full B–F migration acceptance is claimed.

| Evidence epoch | Result and scope |
| --- | --- |
| Earlier dual183 selection, per stack | 183 cases / 549 phases PASS, zero test failures/errors/skips: all 172 original IDs plus four Asin and seven pure-data corruption controls |
| Dual183 export attempts, per stack | 140 positive returned exports and two expected-negative attempts; 142 facade attempts in total |
| Dual183 native observations, per stack | 293 observed helper attempts, 291 successful snapshots and 985 retained native arrays; these are observer units, not a process-wide evaluation census |
| Dual183 CPU target execution, per stack | 346 successful sessions / 366 successful runs and 1,458 retained target arrays |
| Dual183 sample reports, per stack | 341 reports / 361 samples: 352 VALIDATED and nine FAILED; eight original Sign failures plus one intentionally corrupted target |
| Dual183 direct calls outside reports, per stack | Four Gather runs and one original scalar run independently checked, including both native reference roles |
| Retained denominator | All 318 complete elementary raw contracts / 636 caller obligations; all 1,852 inventory rows / 3,704 caller obligations and 50 optional model obligations unchanged |
| Earlier style bridge | Pinned Black25/Ruff PASS on the then-touched 72 Python files; complete AST unchanged for the single formatted test, 1,085 other inputs and production held; this precedes later reviewed repairs |
| Final focused15 v2, per stack | **1,937 cases / 5,811 phases PASS**, zero failures/errors/skips; independent actual-evidence acceptance |
| Final complete PRIMARY v4 | **10,237 cases / 30,711 phases PASS**, zero failures/errors/skips; all four worker schedules match the exact released collection/controller observations |
| Final local code qualification | **`11748451`**, 86 exact Python blobs, all seven normal code hooks PASS, 923 Python / 163 other pins held through hooks/commit, clean worktree at commit verification |

### Implemented finite behavior and retained limits

Native elementary/reduction dispatch and owning typed lowerers now preserve
their reviewed operation/parameter/dtype/weakness/shape contracts. The scope
includes bounded floating elementary functions, numeric/Boolean reductions and
explicit integer value controls; schema/admission evidence and numerical
acceptance remain separate. See the actual owning
[admission rules](../../../jax2onnx/converter/source_elementary.py),
[lowering algorithms](../../../jax2onnx/plugins/jax/lax/_source_elementary.py),
[finite selection](../../../tests/extra_tests/converter/test_source_elementary_reductions.py)
and [Asin controls](../../../tests/extra_tests/converter/test_source_asin_repair.py).
All original callables, feeds, source contracts, defaults, opsets and criteria
remain frozen. No tolerance or FTZ/runtime-policy change is introduced.
Generic SOURCE admission currently accepts explicit opsets **16, 18, 21, 23,
24, 26 and 27**, with operation/type-specific eligibility; this does not certify
every operation/type at every admitted opset. Product `scan`/`while` admission has
a separately frozen external candidate and is **not integrated**: five existing/
two new paths, **121 forecast cases, not collected**. Frozen-byte/runtime/typing/
dual qualification is pending. This prepared implementation checkpoint follows
original-six follow-through; current source-loop evidence uses diagnostic admission.

Asin FP32/FP64 uses same-width rational coefficients and a half-angle expression,
adapted from the reviewed FDLIBM/OpenLibm formulas. This is not a claim to the
upstream implementation's full correctly-rounded contract. The original FP32
pointwise/consumed-sum witness now passes; the consumed sum is exact positive
zero against both native references and all four target results. Enriched
domain/signed-zero controls and new `(B, 33)` region/lane-consumer feeds at B=2/5
pass their unchanged finite criteria on both stacks. Coefficients, association,
widths, domain and sign behavior retain explicit structural checks.

The two original Sign diagnostic tests pass by retaining **eight FAILED
numerical reports per stack**. At the smallest positive subnormal, native
output/reciprocal words are +0/+Inf while target words are +1/+1; all unaffected
array words remain checked. The seven pure-data controls reject other
corruptions. This is neither Sign numerical acceptance nor proof of ONNX
impossibility. The earlier failed strict Sign experiments remain history.

All **44 original composition artifacts remain numerically UNVERIFIED**, with
raw callbacks **30 FAILED / 14 PASSED**. Both original trigonometric Gather raw
callbacks remain **FAILED** and raw SOURCE criteria remain None. Their separate
finite numeric controls pass both frozen predicates against both eager/fresh-JIT
roles and both target modes; neither predicate replaces the other. Passing the
183 tests therefore does not close all 318 raw obligations.

Earlier Erfc limits remain: graph/schema evidence at opsets 23/26, configured CPU
numerics only at opset 23, strict full-seven-output/native-agreement gates
**FAILED**, FP16/BF16 numerics **UNVERIFIED**, and no activated whole-carrier MLIR
claim. The four GPT_MLP caller/stack cells now also have independently reviewed
actual focused15-v2 evidence: original **768 → 3,072 → 768**, FP32 model/state,
seed, zero Dropout and approximate GELU, under the unchanged strict **1e-5**
criterion. The actual body retains typed `(x*x)*x`, the original coefficients,
second Linear and key-payload provenance. Saved native/target arrays carry the
bounded numerical acceptance. Export reports separately keep runtime and
numerics **UNVERIFIED**; they do not inherit sample acceptance. This finite
GPT result does not close the 44 original composition obligations.
Optional 45 MaxText/five MaxDiffusion obligations stay **AUDIT_PENDING**;
secondary MaxText Python >=3.12 and the separately recorded missing tokamax
prerequisite remain environment work, not converter acceptance.

### Retained author and preflight epochs

The detached original v4 epoch remains **169 PASS /three target-sample FAIL
of 172**, and the later v6 Asin/Sign experiment remains **one PASS /six FAIL
of seven**. The earlier checkpoint below retains their exact observations and
saved-byte analysis. Neither is rewritten by later source improvements.

The v8 author12 epoch (retained under its `v7-author` directory name) finishes
**10 PASS /two structural FAIL**, 36 phases. The original/enriched three Asin
numerical cases and seven pure-data controls pass. The new symbolic FP32/FP64
artifacts were exported and saved but stopped before target comparisons because
a global arithmetic-width assertion also examined unrelated INT64 reverse-index
Sub nodes. Offline ancestry proof identifies every offender outside Asin's
output cone. A narrowly reviewed checker-only repair retains all owning Asin
width/coefficient/association/domain/sign checks. The separately released
symbolic retry then passes **two cases /six phases**, with unchanged product
bytes/native captures, eight sessions and 16 runs. Author evidence is three
Asin numeric cases plus the separate two, never a fabricated 12-case PASS epoch.
The later actual integrated dual183 executions are distinct new epochs.

The first SECONDARY combined183 preflight fails the origin guard before
collection: **zero cases, zero artifacts**, no model/target execution. Its old
editable PEP420 `scripts` namespace is a harness-origin failure, not an ONNX
feasibility result. The reviewed v3 namespace bootstrap changes no source,
dependency or environment package. The primary v2 epoch is retained unchanged;
a freshly released secondary v3 epoch passes all 183 cases. Both failed epochs
and their terminal/source pins remain preserved.

### Actual regression failures and final qualification

The first complete PRIMARY execution (`full-v1`) finishes **10,204 PASS /
26 FAIL of 10,230 cases**, zero errors/skips. It remains a failed epoch. The
subsequent `full-v2` and `full-v3` packets perform **collection only** and are
never claimed as full executions.

The first both-stack focused14 execution finishes **1,879 PASS / six FAIL of
1,885 cases per stack**. After reviewed fixture/capture changes, focused15-v1
finishes **1,931 PASS / six FAIL of 1,937 cases per stack**. Its failed key-array
weakness capture is an actual failure; GPT numerics were not reached in that
epoch. A separately reviewed **one-line** diagnostic correction reads
`jax.typeof(value).weak_type`, preserving typed-key payload provenance, other
source bytes, original models and criteria.

The resulting focused15-v2 execution passes **1,937 cases / 5,811 phases on
each supported stack**, with zero failures/errors/skips. Independent acceptance
binds the actual terminal receipts, every phase, saved artifacts/arrays, original
GPT model and strict comparisons, and the still-failed Sign reports. The actual
complete PRIMARY v4 then passes **10,237 cases / 30,711 phases**, with zero
failures/errors/skips; all four worker schedules match the released collection
and controller observations. Full-v4 uses the
explicitly released CPUs 8–11 for collection and CPUs 4–11 for execution.
All **923 Python / 163 other input pins** hold.

Independent readiness review accepts the finite focused/full code checkpoint.
Actual normal code commit **`11748451`** then passes **all seven normal hooks**,
with exact 86 committed Python blobs, all qualified input bytes held through
hooks and commit, and clean worktree at commit verification. The earlier
`1533d60b` full/hook result remains acceptance of its preceding source epoch.
Passing regression diagnostics do not reclassify any known numerical failure.

### Historical original-six capture and then-planned gates

This capture-only record precedes the saved-artifact structural and sample
continuation above. Its failure, zero-target-execution counts and then-open
expansion gap remain historical evidence.

The fresh original-six follow-through uses the qualified post-elementary
immutable snapshot. ROOT's release binds the original six IDs and unchanged
model source/state/criteria. **Both-stack collections PASS with six exact IDs
each and zero factory/native/trace/export/ORT calls**; collection alone is not
model qualification.

The actual first PRIMARY cell (`onnx_functions_014` ordinal 0, caller x64 off)
finishes **one test FAIL, zero errors/skips, terminal exit 1**. The export itself
**COMPLETES**: one ONNX artifact passes schema/ports validation and one body
manifest is saved; one export returns and zero raise. All **32 scheduled native
evaluations and their 32 outputs** are captured before assertions, with **two
restored-native evaluations** and **47 NPZ archives** retained. Source, maps and
import origins hold. **Zero ORT constructors/sessions/runs** occur.

The unchanged original raw graph callback remains **FAILED**: root inputs
`in_0`, `in_2`, `in_3`, `in_4` and `in_5` are unused, and the expected top-level
`SuperBlock_1:5x10x3` path is absent while the root contains If.

Offline review verifies a **wrapper-scope mismatch**, not dropped inputs. Root
`If(in_1)` branch GraphProto bodies lexically capture `in_0`, `in_2`, `in_3`,
`in_4` and `in_5`; each branch calls SuperBlock. The checker only examines
direct root nodes. The original six parameter payloads match their recorded
formal/state positions. Native bodies retain LayerNorm → 3-to-6 Linear →
Erfc-based exact GELU. The true branch forwards counts/keys; the false branch
increments each count and preserves keys.

The completed bounded composition review finds no dropped-input, swapped-
parameter, dead-mask, crossed-stream or state-feedback mismatch. All **30
deterministic / 42 stochastic Function equations** are independently proved
alpha-equivalent to the **unpatched prepared trace**, including typed operands,
literal hashes, nested Bernoulli/uniform operations and output/count/key
positions. In the serialized ONNX, all six parameter positions are exact.
Mask 1 uses only count1/key1 and selects zero versus GELU/0.9 before the second
MatMul. Mask 2 uses only count2/key2 and selects zero versus second-Linear/0.9
before output. Each stream has its own count Add and unchanged key; deterministic
state is forwarded. These are bounded composition/position proofs, not a full
expansion-level equivalence proof for every serialized node.

Full **BODY_FIDELITY remains UNVERIFIED**: complete Threefry/bitcast/Erfc
expansion equivalence across the **1,725 serialized stochastic nodes** was not
reproved. **Target numerics remain UNVERIFIED**, with zero target execution.
Keep the original raw callback **FAILED**, specifically for its direct-root/
lexical-If blind spot. This external follow-through does not change separately
accepted code `11748451`.

The capture-first protocol retains its original **35-native/two-trace/one-export
first-cell ceiling with zero ORT**. Next, perform a bounded **offline expansion-
correspondence check** using the saved artifact and already-qualified primitive
lowerings, then review a scoped wrapper-aware structural counterpart. Preserve
the raw callback failure, original models and criteria. These checks require no
new trace/export, tolerance change or ORT execution and establish no continuation
or target acceptance by themselves. Remaining cells and complete-model
acceptance stay open.

Continue all remaining original-model, normalization graph-contract,
fixed-source preparation, optional-model, low-precision and scientific FP64
work. All B–F gates and the final ordinary source-preserving public UX over
**all existing functionality and fixed third-party sources** remain required.
A permanent mode/profile selector, source substitution, per-operation LEGACY
fallback or loss of existing capability does not complete the migration.
Public behavior stays **LEGACY**.

Canonical local receipts (provenance, not tracked execution dependencies):

| Evidence | Receipt | SHA-256 |
| --- | --- | --- |
| Actual ROOT dual183 proof | `/tmp/jax2onnx-elementary-root-integration-v1/root-actual-dual183.json` | `43abfe3e8812e7b6f4563a09d5fbefb7105d1ec5b5d58461ca7d581d94c28a35` |
| Independent actual dual183 acceptance | `/tmp/jax2onnx-astra-dual183-evidence-review-v1/review-dual183.json` | `577c0a7b4462865b49607fc221380b7906ae6a73e97ae145d74ab2c3ac252e4c` |
| Independent direct Gather/scalar call checks | `/tmp/jax2onnx-astra-dual183-evidence-review-v1/direct-target-calls-review.json` | `1159c7c8767c6da7e9c45c96629a5371648a989c67bf7f120b706d12c6a33685` |
| Exact65-source application | `/tmp/jax2onnx-elementary-root-integration-v1/root-application.json` | `c594c3290cd5025687ca0d60d7187a4464c9989060e147b00fa4b2d3531998e3` |
| Later seven-file scope-test application; production unchanged | `/tmp/jax2onnx-elementary-root-integration-v1/root-scope-test-repair-application.json` | `2cbf19f3ecfbe4face11a6c3f04001f6ab852982c86680d0c31b99ad12979354` |
| Retained failed author12 terminal | `/tmp/jax2onnx-source-elementary-v7-author-primary-root-v1/root-actual-terminal.json` | `d13929124597494738932e81594ca8dc6ac2870151c45905baecef66e1b622f6` |
| Independent author12 saved-byte review | `/tmp/jax2onnx-astra-symbolic-asin-structural-review-v1/actual-author12-review.json` | `01a0475757669a47533777e1c90543cebf87324e71b2d2bb65284d5dec5e43b3` |
| Narrow structural repair and retry2 preparation | `/tmp/jax2onnx-symbolic-asin-structural-repair-v1/MANIFEST.json` | `12559986a2f0626f4ceecbb3e520a62ee1f091f1713478bcd93c3452211611d0` |
| Actual symbolic2/aggregate-five author review | `/tmp/jax2onnx-astra-symbolic-asin-structural-review-v1/actual-symbolic2-and-aggregate5-review.json` | `afee63d6f6fb5784719e5aa1390bec60030d62e44d6a44f5c7eee37bf9452c02` |
| Retained failed secondary preflight | `/tmp/jax2onnx-elementary-root-integration-v1/combined183-secondary-v1/root-actual-terminal.json` | `9f898ca133e7a8a2ef0184e8b7b7a248f53a61466f152cd94dd42e59c69ded51` |
| Earlier original-six static preparation only | `/tmp/jax2onnx-original6-astra-preparation-review-v1/review-v2-preparation.json` | `ed226b1f4a080061cba36c3cae6ae059e7a952557256ec7d133134dfb6c045f9` |
| Pinned repository-config formatting bridge, AST identical | `/tmp/jax2onnx-elementary-root-integration-v1/root-pinned-format-proof.json` | `64e2f7fe3f0f738aa0d6d1d2105fce95f918f79e96b8e2c25ef92c4a15b52b52` |
| Actual pinned Black25/Ruff72 PASS, separate from normal hooks | `/tmp/jax2onnx-elementary-root-integration-v1/root-style72-pass.json` | `cef09361a3fdfe0732beb6a0bea5ea25b37d2d6be97fbab2b16983bcb34beec7` |
| Actual final code commit and normal hooks | `/tmp/jax2onnx-elementary-root-integration-v1/root-code-commit-11748451.json` | `2484a32bf442fcffb22692da11764526619f554779ce936015f302db1a747b74` |
| Independent actual dual focused15-v2 acceptance | `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-dual-focused15-v2-acceptance.json` | `b0b664c821bc9eeaf73f3177dc868d6e8ceb13a90b4f10ca0df54088327bd42f` |
| Independent actual complete PRIMARY-v4 readiness | `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-full-primary-v4-code-checkpoint-readiness.json` | `449e6c5b4afcc18be5366dfca9aa6643aa8b8f9dc5e082a739bde91983a2487d` |
| Independent original GPT bounded evidence | `/tmp/jax2onnx-astra-fixture-key-review-v1/actual-focused15-v2-original-gpt-bounded-evidence-review.json` | `1eeab1f73baac783555b5de106a202108128789e99c834a6ea87b6099dae0a73` |
| Retained failed full-v1 actual terminal | `/tmp/jax2onnx-elementary-root-integration-v1/checkpoint-full-primary-v1/root-actual-run-terminal.json` | `29af615fc973bfb72900f0dfa19f8c85e1fe92cf674659c8b59318184ef63ffa` |
| Retained failed focused14-v1 summary | `/tmp/jax2onnx-elementary-root-integration-v1/root-actual-dual-focused14-v1-failed.json` | `f5adfe7c846ddb3daafc42cb3f4cf9f1ae167271c8d3217a3f1e39f93579b8f7` |
| Retained failed focused15-v1 summary | `/tmp/jax2onnx-elementary-root-integration-v1/root-actual-dual-focused15-v1-failed.json` | `684f7c9be7fc8a7f424da6cd70435e8a34b8b5ad1a482525ba32aa391f149ff0` |
| Actual one-line key weakness correction | `/tmp/jax2onnx-elementary-root-integration-v1/root-key-array-weakness-v1-application.json` | `aeedbd8feece5c32bfc6613c55b78bfb974ea5ac20c42a878ac46704a4e98f6b` |
| Original-six bounded fresh release | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/release.json` | `2cd22f65d84895c34d56bc7e4854ed28ffa470dce1342f40a0ddb2fc7a85cfc0` |
| Original-six actual dual collection only | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/root-actual-dual-collections.json` | `138d6c962b8f47ef22426ee258fa3058df6fee7a8eef652bd3933e577cd051ea` |
| Original-six actual first terminal, export complete / raw callback FAILED | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/root-actual-first-terminal.json` | `34bcdb8a650ff50da38c637a54568595f71cbbc0f83404f0b2f51cf18de19bec` |
| Original-six actual first raw receipt | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/results/primary/first/receipt.json` | `d8e0560b444637b5731bbe3ea142819915cff56281c7dc76d0cf8b8fb5fe19c1` |
| Original-six actual first log | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/results/primary/first.log` | `dbe4224c0c79609099a0a1374251bd07f4d65089d13f66dab7e162c53fa9fd00` |
| Next external loop-admission candidate, not current implementation | `/tmp/jax2onnx-source-loop-admission-candidate-v1/MANIFEST.json` | `1f7a6481ad06240200e0bd3671bca9a3597afe920df3c09d8a5cb93ea40e60f0` |
| Original-six actual saved ONNX artifact, no target execution | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/results/primary/first/tmp/test_original_normalized_model0/explicit-state/source.onnx` | `1e67c8d21e34d26522892b751cab6fa042f819f6611060d16f5c060e1cf773c6` |
| Original-six actual source trace | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/results/primary/first/tmp/test_original_normalized_model0/explicit-state/source-trace.json` | `20333feef7aa6859c4606c98dbaddd4b0535c8113875fef827fd15ff6600ea17` |
| Original-six observed native source traces | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/results/primary/first/tmp/test_original_normalized_model0/explicit-state/observed-source-native-traces.json` | `6ea28cf4d37cbe983c0777b880f753646808991c711bae890b7db9ad4c06715f` |
| Original-six saved artifact body manifest | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/results/primary/first/tmp/test_original_normalized_model0/explicit-state/artifact-fidelity.json` | `f83f2805ff93e8f53019f5cc7ae44120fda2c8fb647196bdfea1ff304a5470ea` |
| Original-six unchanged raw callback FAILED | `/tmp/jax2onnx-elementary-root-integration-v1/original6-after-elementary-v1/results/primary/first/tmp/test_original_normalized_model0/explicit-state/raw-callback.json` | `677c3e47e4351b4103b19e1433450ced351bd9e840c5e421895ed4fdc35e8102` |

## Tensor index, Erfc and optional inventory checkpoint

Status: **ACCEPTED LOCAL CODE CHECKPOINT — `1533d60b`, all seven normal code
hooks, actual full-primary and separate dual-stack finite PASS evidence, with
independent final code-evidence review ACCEPTED**. The prior ROOT selection
retains **786 cases / 2,358 phases PASS per stack** on unchanged production. The separate final-byte test-repair selection passes **29 cases /
87 phases on each stack**, zero failures/errors/skips and held current **919
Python / 163 other inputs**. The independent review accepts that bounded test
repair for ROOT integration/qualification; it is not full milestone acceptance.
Earlier accepted code checkpoint `39ddee8c` remains recorded in the next section.

| Checkpoint evidence | Accepted scope |
| --- | --- |
| Local code commit | **`1533d60bb82c3860e7098221dc6b9abc68562c0a`**, exact 40 committed blobs match the full-qualified sources; tree clean at commit verification |
| Retained ROOT dual-stack selection | **PASS: 786 cases / 2,358 phases per stack** on unchanged production; failed focused epochs remain separate |
| Separate final-byte test-repair qualification | **PASS: 29 cases / 87 phases per stack**, zero failures/errors/skips; current 919 Python / 163 other input pins held |
| Complete PRIMARY suite | **PASS: 10,047 cases / 30,141 phases**, zero failures/errors/skips; four matching collections/controller observations and held current 919 Python / 163 other inputs; earlier interrupted/failed runs remain history |
| Secondary milestone qualification | Retained **786-case / 2,358-phase PASS**, plus separate **29-case / 87-phase PASS**; no complete-secondary suite claim |
| Fresh final-source quality | Pinned Black25/Ruff and diff check PASS on 33 files; actual typing PASS on 547 source files plus RNG report; these are separate from normal hooks |
| Normal code commit hooks | **PASS: all seven hooks**, zero skips, exit 0; all 919 Python / 163 other input bytes unchanged |
| Final independent code-evidence review | **ACCEPT**: actual HEAD, all 40 committed blobs and all 1,082 qualified inputs, full PRIMARY result, separate dual29 and retained786 bridge, seven normal code hooks and retained limits verified |

### Actual focused result and retained failures

Each initial ROOT selection collected **779 cases / 2,337 phases**, with
**770 passes / nine failures**, zero errors and zero skips, on both supported
stacks. The independently checked passing implementation subset is **674 cases
per stack**:

| Selected test module | Passing cases per stack |
| --- | ---: |
| Tensor/index families | 296 |
| Gather | 99 |
| Shared symbolic batch | 43 |
| Representation | 136 |
| SOURCE profile | 52 |
| Erfc | 46 |
| Original GPT_MLP numerics | 2 |

The nine failures remain failed history: one inventory test fixture discarded
five legitimate zero-case registrations, and eight old corpus admission checks
used a stale whole-oracle descriptor pin. The external three-file repair
preserves unrelated registrations and the exact historical corpus controls,
updates only the explicit v5 revision and current descriptor digest, and adds
five corruption cases. Its bounded PRIMARY author smoke passes **14 cases /
42 phases**, with zero failures/errors/skips, CPUs 12/13, one thread and no ORT
sessions. All project imports resolve to its detached source snapshot and source
pins hold. This is author evidence; the original 779-case outcomes are not
rewritten as passes.

The subsequent ROOT **784-case** selection is also terminal on both stacks:
**783 passes / one failure**, zero errors/skips, with held source pins. The one
remaining failure is a stale current Gather test consumer asserting v4 after
the reviewed v5 control revision. This epoch is distinct from the original
779-case/770-pass/nine-failure result.

A separate three-test-file consumer supplement updates the two current v5
assertions, preserves all historical v1/v2/v3 rejects and adds v4 as a negative.
The current first-selection test now asserts 1,852 rows / 3,704 total / 36
selected / 3,668 unselected caller cells, freezes the original 36 identities and
retains their None-criteria check; all 100 optional caller contexts are proved
unselected. Its PRIMARY author smoke passes **three cases / nine phases**, with
no ORT and only the existing four Add/dot control artifacts. Production code,
authentic historical 1,802/3,604/3,578 corpus guards and prior evidence assets
are unchanged by this supplement. The subsequent final ROOT selection passes
**786 cases / 2,358 phases on each stack**, with zero failures/errors/skips and
held sources. It combines the retained 674 implementation cases with 112
inventory/replay cases; the earlier failed epochs remain unchanged.

The earlier ROOT complete-suite **collection** succeeds with **10,047 distinct
identities** and held sources. Relative to the preceding 9,667-case checkpoint,
381 identities are added and one obsolete unsupported-Erfc negative
(`test_source_profile.py` unsupported case `[lambda4]`) is intentionally removed;
all other old identities are retained. Collection is not test execution; the
separate final complete-primary result is recorded below.

Actual prepatch package typing passes **547 source files plus RNG registration**.
Pinned Black/Ruff checks pass on the 24 checkpoint files. These do not replace
the final full-suite result or the separate code-commit hook evidence below. The independent
actual-result review binds **919 Python / 163 other inputs** and verifies both
focused collections, every phase, XML and import provenance. Passing tests may
record failed numerical gates; 674 passing cases are not broad CPU acceptance.

### Interrupted and failed complete-suite epochs

The first full-primary attempt ended with **exit 120 during ENOSPC**, with
source/input pins held. Four worker collections each contained 10,047 cases,
but the XML and process receipts were truncated; a complete phase/XML result
is **NOT_AVAILABLE**. One test failure was observed before disk exhaustion,
with details not flushed. This is an interrupted epoch, not a completed
full-suite result; its partial evidence remains preserved.

The subsequent complete-primary retry finished all **10,047 cases / 30,141
phases** with **10,038 passes / nine failures**, zero errors/skips, equal worker
collections and held sources. These are nine stale finite-profile exclusions
or current-denominator assertions, not reasons to remove newly audited product
behavior. The failed full result remains unchanged.

The independently accepted five-file/six-function test repair preserves every
original parameter ID, callable, criterion and genuine negative. Integer Add
checks exact modular overflow; native arange/linspace are checked directly and
through consumers; valid reshape admission is tied to the passing common CPU
controls. The current 24-cell numerical selector retains its original identities
and None criteria, with 3,680 unselected cells and all 100 optional contexts
unselected. Production sources and authentic historical guards/assets remain
unchanged. Its author smoke passes 29 cases / 87 phases; the import-only
post-smoke cleanup is explicitly bound by the independent review.

ROOT subsequently applies and qualifies the exact final test bytes on **both
stacks: 29 cases / 87 phases PASS per stack**, zero failures/errors/skips and
identical current 919 Python / 163 other input pins held. The five changed test
files are not selected in the retained 786-case execution and production is
unchanged, so that prior finite result retains its exact scope. These are
separate executions; no single 815-case execution is claimed.

The fresh full collection contains the same **10,047 identities**. Final pinned
Black25/Ruff checks on **33 files**, diff check and the freshly executed package
typing check (**547 source files plus RNG registration**) pass on held sources.
These do not replace normal commit hooks.

The final complete PRIMARY execution then passes **10,047 cases / 30,141
phases**, with zero failures/errors/skips, terminal exit 0 and **677.54 seconds**
elapsed. All four worker collections match each other and the controller
observations; the current **919 Python / 163 other input** pins hold. This
actual full result preserves both earlier failed epochs and the separate
dual-stack finite scopes above.

The subsequent normal local code commit is
**`1533d60bb82c3860e7098221dc6b9abc68562c0a`**. All **seven normal code hooks pass**,
with zero skips and terminal exit 0. The exact **40 committed blobs** match the
full-qualified sources, all **919 Python / 163 other input** bytes remain
unchanged through hooks and commit, and the tree is clean at verification.
Independent final code-evidence review **ACCEPTS this finite local checkpoint**
and verifies the actual HEAD, all 40 committed blobs and all 1,082 current
qualified inputs, the full PRIMARY result, separate dual29 result, retained786
bridge, all seven normal code hooks and the failed/unverified limits below.
The tree is clean at that review. This is a local code checkpoint with no push
or public transition; the review deliberately does not bind subsequent
documentation postimages.

### Current common tensor and index scope

The finite selector freezes **63 original rows / 126 caller-x64 identities**
across concatenate, dynamic slice/update, gather, iota, pad, reshape, reverse,
slice, squeeze and transpose. The 296-case module passes on both stacks;
**56 selected rows / 112 tested caller contexts** contain **109 positive
artifact contexts and three native/configuration negatives**. Gather ordinals
2/3 under caller-x64-off retain their FP64/configuration conflicts, and
`dynamic_update_slice` ordinal 5 under caller-x64-on retains the original-source
INT64/INT32 index-type error. Those three tests pass by recognizing the original
negative contracts; they are not positive artifact admissions. The remaining
seven rows stay deferred: gather ordinals 0/1 (trigonometric composition), pad 3/4
and slice 2 (scan compositions), reshape 5 (multiple symbolic dimensions) and
reshape 8 (derived `B*4` extent). No omission or diagnostic counterpart closes
those product obligations.

Shared-B dimension values preserve native weak INT32/INT64 behavior under the
effective local x64 setting, including settings opposite to the caller. The
reviewed unsigned-index path uses an exact local INT64 bridge rather than a
lossy narrowing cast; tested large unsigned and signed limits retain the native
clipping/wrapping contract. Squeeze, pad, copy and related tensor paths retain
the checked source types, shapes and consumed values. These are bounded forms,
not general symbolic/index/control-flow acceptance. Original factories,
criteria, opsets, callbacks and the complete 63-row contract aggregate remain
unchanged.

### Current Erfc representation and numerical evidence

The private SOURCE route has Erfc **graph/schema evidence at opsets 23/26**,
preserving the reviewed FLOAT/DOUBLE expression, source dtype, shape and
weakness. Configured CPU numerical evidence is **opset 23 only**: the artifact
helper and diagnostic worker use that default. Opset 26 CPU numerics remain
**UNVERIFIED**. FP16/BF16 have structural representation/castback evidence only; their numerical status
is **UNVERIFIED**. No production FTZ initializer, target configuration or public
runtime-policy change is introduced.

The configured diagnostic completes **36 native evaluations, four exports,
eight sessions and 12 runs per stack**. Offline verification of the final focused run
reproduces the saved-byte summary with unchanged counts and **zero misleading
zero absolute-error metrics**. The independent earlier offline comparison is
retained separately. Measured direct Erfc,
ordinary-composition and named shared-batch outputs pass their fixed gates,
including 2,004 direct, 1,200 ordinary and 1,008 named scalar comparisons per
stack. These totals describe comparisons within those finite scopes.

The literal **full-seven-output gate remains FAILED**, and **native reference
agreement remains FAILED**. Of 10,068 strict fixed comparisons per stack,
48 fail on the `t` / `half_x` prefix values. Native eager and JIT already disagree
on those FP32/FP64 subnormal values, and each recorded target word equals one of
the conflicting native words. No tolerance, sign/class requirement or output
role is relaxed or discarded; zero-valued absolute-error metrics do not hide
these word/class failures. The old default-CPU fixed-grid failures remain
**FAILED**, separate from the configured environment. Named activated
whole-carrier runtime remains **NOT_ESTABLISHED**.

The earlier six-Function-model run reached Erfc in its first cell, with one
failed stale-barrier assertion and five unexecuted cells per stack, no returned
model artifact and no ORT attempt. That observation belongs to the previous
code. A fresh original-model run is needed after the new lowering; no downstream
barrier or complete-model result is established here.

### Original GPT_MLP FP32 scope

The unchanged original GPT_MLP ordinal-0 contract now has finite numerical
**PASS** evidence on both stacks in **two FP32 caller-x64 contexts**. The real
model, parameters, Boolean runtime control and declared FP32 feature-port
contract are retained. The criterion is the reviewed generated FP32 default for
these fixed samples. This advances the earlier artifact-only observation; it
establishes neither FP64 numerics nor the other original model contracts,
training/stochastic equivalence or general complete-model coverage.

### Inventory and optional environment matrix

Inventory SHA-256:
`b875cf3a8169c2919c425eb9d5f11fda9396020945d7ffc4164fb748e153474c`.
The denominator is **1,852 rows / 3,704 caller obligations**: all **1,802 original
core rows / 3,604 caller obligations**, with exact canonical registration,
ordinal, name, feature bits, opset and complete contracts, plus **45 MaxText /
five MaxDiffusion** retained obligations. The manual inventory has **234 files /
1,565 functions**, retaining all **752 original anchors**. Retention is not
SOURCE acceptance; all 50 optional source/model obligations remain
**AUDIT_PENDING**, with target numerics **UNVERIFIED**.

The six original inventory-only CLI attempts are terminal **four PASS / two
FAIL**. Each receipt records one observation, held source pins and no native
model, exporter or ORT entry calls.

| Stack and environment | MaxText / MaxDiffusion observed rows | Actual result |
| --- | --- | --- |
| PRIMARY, unconfigured | 0 / 0 | PASS; reviewed optional baseline obligations retained |
| PRIMARY, defaults | 6 / 1 | PASS, inventory only |
| PRIMARY, all | 45 / 5 | PASS, inventory only |
| SECONDARY, unconfigured | 0 / 0 | PASS; documented `one_minus_square` version absence retained |
| SECONDARY, defaults | Explicit `Environment_Error` observations | FAIL, nonzero exit |
| SECONDARY, all | Explicit `Environment_Error` observations | FAIL, nonzero exit |

The pinned MaxText revision
`17d805e3488104b5de96bd19be09491ff73c57c1` declares **Python >=3.12** and uses
syntax that fails on the Python 3.11 secondary stack. This is an upstream
prerequisite, not a proved SOURCE/ONNX limit. Core Python 3.11 support remains;
no upstream ref/source or dependency-policy change is made to manufacture a
configured import pass.

The initial MaxDiffusion error was missing `PIL.ImageOps`. A bounded secondary
environment preparation installed only the declared locked **Pillow 12.3.0**
wheel, verified by SHA-256
`23d27a3e0307ec2244cc51e7287b919aa68d097504ebe19df4e76a98a3eea5bd`.
Repository source, dependency policy and upstream refs remain unchanged. The
supplementary SECONDARY **MaxDiffusion-only defaults** CLI then fails on the next
actual missing dependency, **tokamax**, with nonzero exit, one observation,
held pins and no forbidden model/export/ORT calls. The supplementary
**MaxDiffusion-only all mode is NOT_EXECUTED**. No further environment repair or
secondary configured positive is claimed. Preserve the initial six attempts
and this supplementary failed epoch separately.

### Remaining checkpoint and migration work

The current finite code checkpoint is independently accepted. Astra has
separately **accepted the bounded v3-to-v4 static delta for ROOT validation** of the separate elementary/
reduction candidate. Its packet has **63 source files, 27 test functions,
172 declared cases and 318 selected raw rows / 636 caller obligations**. It
remains **unintegrated and not ROOT-qualified**. Its detached PRIMARY author
epoch executes all **172 cases / 516 phases**, with **169 passes / three
failures**, zero errors/skips and held source pins. Its **44 original composition
diagnostic artifacts remain numerically UNVERIFIED**; raw structural callbacks
record 30 failures / 14 passes, and both Gather raw callbacks fail with their
original None criteria unchanged. The 169 passing pytest cases do not establish
all-source numerical acceptance. The failures are the Asin
FP32 consumed sum (the pointwise values meet the fixed criterion) and Sign
FP32/FP64 smallest-subnormal CPU mismatches. These are actual failed target
samples, not a demonstrated ONNX impossibility. Original model source, input
bytes, defaults and criteria remain frozen; no FTZ or tolerance change is
introduced. A frozen three-case diagnostic plan preserves the exact failed selectors; fixes remain separate
owned work. A later detached seven-case Asin/Sign experiment finishes with
**one pass / six failures**: the enriched FP32 Asin control passes, the original
Asin sum still fails, and the FP64 Asin control encounters the missing ORT
`Asin(22)` DOUBLE kernel. Original and extended Sign checks fail; native
eager/JIT ordered comparisons disagree for subnormals. This is engineering
evidence, not a MAIN regression, fix acceptance or ONNX-model impossibility.
Frozen saved-byte analysis confirms the failed proposals remain unaccepted
(`/tmp/jax2onnx-source-elementary-reductions-seven-experiment-primary-v2-offline-analysis/analysis.json`,
SHA-256 `ada90aa959e40f8935c0eb9d3e2f91cfe85905722059ecfe5d7c7139094a022b`).
Qualify the candidate separately after this code checkpoint. Its source files,
selector rows and declared tests are not milestone product behavior or
migration/numerical passes.

Continue unchanged original-model follow-through, native normalization graph
contracts, whole-carrier MLIR, all remaining registered functionality and general
fixed-source preparation. All B–F gates remain required, including FP16/BF16
rounding, selective FP32 and scientific FP64. No partial private profile can
complete the task, no per-operation LEGACY fallback closes a row, and no permanent
mixed public precision interface is planned. Public behavior remains **LEGACY**.

Canonical local receipts (provenance, not tracked execution dependencies):

| Evidence | Receipt | SHA-256 |
| --- | --- | --- |
| Independent actual-result/environment review | `/tmp/jax2onnx-precision-milestone-actual-results-astra-review-v2.json` | `d49844b7d6f40f3a2a49c01ed43043d40688def522e881987ad1ef10309c4339` |
| Initial ROOT PRIMARY focus | `/tmp/jax2onnx-precision-milestone-root-v2/focused/primary/root-receipt.json` | `e74a6ad755deb2a4354a118eb352a24700bab230e9428a1f0ceaee01dbbbb54a` |
| Initial ROOT SECONDARY focus | `/tmp/jax2onnx-precision-milestone-root-v2/focused/secondary/root-receipt.json` | `ea482d7b3f7d56978b8be1224b28d6c68b25df243659bac33c47702303f6a4ad` |
| Independent reproduction of Erfc saved-byte comparisons | `/tmp/jax2onnx-precision-milestone-root-v2/focused/root-offline-dual-raw-comparisons.json` | `b7cacb87ff74649ad78a15a3c21fc892b90d04c4fffd04f1bca50cdd62682b1f` |
| External fixture/control repair, with 14-case author smoke and exact source proof | `/tmp/jax2onnx-inventory-replay-repair-candidate-v1/MANIFEST.json` | `fadd72c50dcf56f19430507241a003a8061f678528cf01adfe3f9110f786e97b` |
| ROOT 784-case PRIMARY epoch, 783 pass / one stale-consumer failure | `/tmp/jax2onnx-precision-milestone-root-v2/focused-repaired/primary/root-receipt.json` | `d4b9026ef2796c94c2107b38a0068e471ea2af62ad3d64ae23b7dba8ef17cdce` |
| ROOT 784-case SECONDARY epoch, 783 pass / one stale-consumer failure | `/tmp/jax2onnx-precision-milestone-root-v2/focused-repaired/secondary/root-receipt.json` | `4553fecfb838320a7d10809b2c8a5b478ad7bfb5463bb57ce7d0e4ce77bc2e06` |
| Three current test consumers, with three-case author smoke | `/tmp/jax2onnx-current-test-consumer-candidate-v1/MANIFEST.json` | `e4d3321359faccc59b1c99ebbbfcebde2aaa11630837e69a2e2e7991fd6d70a5` |
| Actual final ROOT focused PASS on both stacks | `/tmp/jax2onnx-precision-milestone-root-v2/root-final-dual-786-pass.json` | `33a093d2747bc0469c73b718ae364c19ac627da2e552fe0f9b988ea6f69bcab1` |
| Final Erfc saved-byte verification, both stacks | `/tmp/jax2onnx-precision-milestone-root-v2/focused-final/root-offline-dual-raw-comparisons.json` | `86105ab857d2322290fa8f8ecca0a5364e40d41f3dcde4417c53b64d516d6819` |
| Elementary/reduction v4 bounded static delta accepted for ROOT validation only | `/tmp/jax2onnx-source-elementary-reductions-astra-review-v4/review.json` | `f5a35f53ff163a7fc9e05776a21ebb893860df580d0eea3a44d7de81826bf345` |
| First full-primary ENOSPC interruption, no complete result | `/tmp/jax2onnx-precision-milestone-root-v2/root-full-enospc-interruption.json` | `321191c0e4ecbd30b762bd3241d3a68e40a606f0faf7a16ba10d249aed7f6d52` |
| Complete full-primary failed epoch, 10,038 pass / nine failures | `/tmp/jax2onnx-precision-milestone-root-v2/root-complete-full-primary-nine-test-failures.json` | `5e3c657738a92f384cb656bb7bb0c9ae8791e51ad4a7ec3d66851b63a6397336` |
| Independent bounded test-only repair acceptance | `/tmp/jax2onnx-stale-profile-full-repair-astra-review-v1.json` | `c8defd807988ce536f262239000bfba338f149b5c705733ff68fe155cb3b1497` |
| Actual ROOT separate final-byte dual29 PASS and retained786 evidence bridge | `/tmp/jax2onnx-precision-milestone-root-v2/root-final-dual29-test-repair-pass.json` | `9ab2777e8ef9f2758eb1da1bc7ca7fb160e7b4ebbb76106a9ba22e31b19318ad` |
| Final pinned style checks on 33 files and diff check | `/tmp/jax2onnx-precision-milestone-root-v2/root-final-style33.json` | `607fd77db756eb70d624526e37f160dc73649cfc98ee706247b47e576d976208` |
| Fresh actual 547-file typing and RNG report | `/tmp/jax2onnx-precision-milestone-root-v2/typing-final-v1.log` | `8c2f34fd5d437e3a0c59e3ebcbff24fe5be4462e05e2b55d4249b63ad29190a0` |
| Detached elementary v4 author epoch, 169 pass / three failures | `/tmp/jax2onnx-source-elementary-reductions-author-full-v4-primary-v1/receipt.json` | `f940821e2bff2dc6286baf85a47116bd73beca314ebd79c046f148394c815873` |
| Frozen three-selector diagnostic plan for the detached candidate | `/tmp/jax2onnx-source-elementary-reductions-failure-diagnostics-v1/plan.json` | `604c82f77921b47df1d231f49e37cf09f132719690d3c687ef128f30d8861e4e` |
| Actual final complete PRIMARY PASS | `/tmp/jax2onnx-precision-milestone-root-v2/root-actual-final-full-primary-pass.json` | `0859886898ef1f2c971ef71dab3d896d4eef659a929dd51fa94482426c796a5c` |
| Detached elementary v4 author summary, retained diagnostic/raw-callback limits | `/tmp/jax2onnx-source-elementary-reductions-author-full-v4-primary-summary.json` | `bb670cb9855876a56c343a8743dd429b6dd7dfac8f7e2693e3bd7778914099dc` |
| Actual local code commit, exact 40 blobs and all seven normal code hooks | `/tmp/jax2onnx-precision-milestone-root-v2/root-code-commit.json` | `db4be722fb2889ea146477e927dbc0dbf38b61c8b1c3a8e053240e1d6b69ad34` |
| Actual normal code-hook log | `/tmp/jax2onnx-precision-milestone-root-v2/code-commit-normal-hooks.log` | `09abdbd7923ef6775bd107a052960f694c944cfb80774aabd03208d0aa60b3ba` |
| Independent final finite code-checkpoint review | `/tmp/jax2onnx-precision-milestone-code-checkpoint-astra-final-review-v1.json` | `eb76c59d611e2f7c555333d82a37dc3b51453d18b74e73cd422efa808dd13912` |

## Joint normalization and matrix integration

Historical checkpoint: all observations and then-current barriers in this
section describe `39ddee8c`, before the tensor/index/Erfc successor above.

Accepted local checkpoint, 2026-10-20: **`39ddee8c5fdf0d4f3a26be53f9e7dd5b0d55c1b6`**,
parent `e8ab9614`. The complete primary suite passes **9,667 tests / 29,001 phases**,
with four identical worker collections, no failures/errors/skips and held source
pins. All seven normal commit hooks pass without bypass. The committed 19 paths,
**913 physical Python files (887 nongenerated)** and **157 other inputs** match
the final tested release; the working tree is clean. This is a local private
checkpoint, with no push or public transition.

| Accepted evidence | Scope |
| --- | --- |
| Options 131 and focused 159, each on both stacks | All 30 original LayerNorm rows / 60 caller-x64 contexts, qualifier controls and exact current facade/corpus/inventory identities |
| Tensor-matrix 44 and retained 348, each on both stacks | Tested static/shared-B matrix compositions, actual exported named bodies, conditional consumers and retained negative contracts |
| Final corrected six, both stacks | Six cases / 18 phases per stack on the final sources, preserving shape-only defaults and diagnostic audit limits |
| Retained secondary selection (721 cases) | 721 cases / 2,163 phases from the earlier exact snapshot; production bytes are unchanged, while four test-file changes and inventory bookkeeping are documented separately |

The manual inventory has **231 files, 1,513 functions and 761 migration additions**,
retaining all 752 original anchors, ten NNX-state identities, 1,802 raw rows /
3,604 caller-x64 cells and 63 opaque records. No full-secondary result is claimed.
Pre-full actual normal typing for **546 files plus the RNG report** was explicitly
reused after checking **574 unchanged inputs** for the assertion-only correction;
that reuse was not a fresh typing execution. The final commit subsequently ran
all seven normal hooks, including mypy, normally.

The earlier full run remains **9,663 pass / four obsolete-control failures**,
with all 9,667 identities / 29,001 phases and no errors/skips. The first corrected
six-case run remains **five pass / one fail per stack**: a new assertion confused
shape-only supplied metadata (`dtype=None`, `weak_type=None`) with the resolved
FP32 default. The minimal assertion correction preserves the original input,
defaults and all other checks. Four old identities map one-to-one to four named
successors; the full count stays 9,667. Both failed histories remain unchanged.

The native zero-reference qualifier is now validated on the fixed original
witnesses: exact eager zero words remain required everywhere, and exact JIT zero
words remain required outside the six pretarget-proven contradictory masks.
The mask cannot depend on target results; reversed signs, tiny nonzero values,
changed inputs or extra conflicts fail. Original sources, defaults, parameters,
opsets, raw tolerances and other dtype/shape/NaN/infinity checks remain unchanged.

All **60 raw normalization graph callbacks still fail**. The accepted evidence
is bounded SOURCE export and numerics for the **30 unchanged callables / 60
contexts under reviewed structural counterparts**. Requested `prefer_native`
records the effective typed-native JAX DAG; it does not establish native ONNX
LayerNormalization emission or satisfaction of the original graph contracts.
Native emission needs a separate eligibility and source-equivalence assessment
for each requested opset and source contract.
Matrix acceptance likewise separates actual exported-body/restored-original
execution from activated whole-carrier ordinary JAX MLIR execution, which remains
**NOT_ESTABLISHED**. Local saved-file counts and test counters do not establish
process-wide native/export/ORT totals.

At this checkpoint the separate original-six exploration advanced the first original
Functions014 cell, caller-x64-off, past the matrix boundary to `erfc` at
`SuperBlock` equation 24 (`profile-validation` / `profile-exclusion`). On each
stack, six cases were collected, **one test failed on its stale matrix-barrier
assertion and five were unexecuted**. Its local records contain **one export
attempt, zero returned model artifacts and zero ORT constructor/run attempts**;
eight native arrays were saved. Neither this first-cell progress nor the
unexecuted five cells establish a complete-model target result. Original sources,
parameters, defaults, exact GELU and live controls remain unchanged.

The first standalone piecewise erfc PoC stopped before ORT on both stacks:
its five original erfc/GELU roles agreed bitwise, but the FP32 consumer gave
eager `0` and fresh-JIT `1`. That failed native gate remains preserved. The
consumer-corrected v3 successor has now executed on **both stacks**, completing
**six recorded ORT sessions/runs per stack**. Native eager/JIT, the five saved
original roles, normal-value and consumer gates pass, but the complete candidate
numerical gates **FAIL** for FP32 and FP64 with optimization disabled and all.
For each comparison to a native reference, the only failures are **13 subnormal
candidate values versus native signed zeros** (five direct erfc, four `t`, two
`half_x`, two GELU); the zero/class requirement remains unchanged.

The configuration-only CPU replay now **PASSES on both stacks**, with exactly
**eight new ORT sessions/runs total** and zero exports, native calls or new inputs.
The same saved FP32/FP64 graphs, seven output roles, eager/JIT references and
unchanged gates pass under `session.set_denormal_as_zero=1`, with optimization
disabled and all. This is finite configured-target numerical evidence. The
**default-CPU gates remain FAILED**; no production FTZ initialization or runtime
policy changes were introduced. SOURCE erfc was still unadmitted at this
checkpoint; the current bounded lowering and retained numerical failures are
recorded above. Low-precision and original whole-model acceptance remained open.

Separately, the exact original GPT_MLP ordinal-0 contract now has a saved,
schema-checked SOURCE artifact with FP32 `B×1024×768` feature ports and the real
scalar Boolean `deterministic` input. This is artifact-only progress: the raw
contract, factory, state/reference recipe and `None` criterion are unchanged;
runtime, numerical acceptance and product support remain UNVERIFIED. The other
21 pinned original parameter contracts retain their first-barrier expectations.

The preceding NNX-only checkpoint and all failed options/matrix/retained epochs
below retain their original scope. General fixed-source preparation, full
models, FP16/BF16 rounding, scientific FP64 and all B–F gates remain open.
Public export behavior remains LEGACY; the complete ordinary-call contract is
unchanged.

The next work recorded at that checkpoint was to assess finite erfc
representation and semantic-lowering scope before proposing SOURCE admission. Runtime acceptance stays tied to the declared
target: the configured pass preserves the known default-CPU failure.
Original model numerics, native normalization graph contracts, whole-carrier MLIR,
fixed-source preparation and the full existing-functionality B–F gates remain
separate work. The exact GPT_MLP contract above remained artifact-only while a numerical
proposal was prepared. Its later finite FP32 acceptance is recorded in the
current checkpoint; no new target acceptance is inferred for this historical epoch.

Canonical local receipts (provenance, not tracked code dependencies):

| Evidence | Receipt | SHA-256 |
| --- | --- | --- |
| Final code commit, exact tested bytes and all seven hooks | `/tmp/jax2onnx-joint-root-checks-v4/root-code-commit.json` | `c855056e00f4692882d04e64898840baba5d92005925e3aef2dce3f3a1615e46` |
| Final full-primary9,667, retained secondary721 and final six provenance | `/tmp/jax2onnx-joint-MAIN-integration-v4/root-actual-final-full-primary-pass.json` | `8f7dc861f9d24e6761beeee88aaa95a1915425fc7dd75478189c6bb1c402859d` |
| Earlier failed full / passing secondary | `/tmp/jax2onnx-joint-MAIN-integration-v2/root-actual-first-joint-full-and-secondary.json` | `d49a589b69b117f08dd650296f6447b8d326e141f02dc2788692db78cd4ad813` |
| Actual standalone erfc zero/class failure on both stacks | `/tmp/jax2onnx-erfc-poc-actual-v2/root-actual-dual-candidate-zero-class-failure.json` | `3b5d9e8f5aee22f2c8d9385d3e22d636474f8d22cb6bf408850778fb54c45dcc` |
| Configuration-only erfc replay recipe | `/tmp/jax2onnx-erfc-denormal-replay-candidate-v1/manifest.json` | `5dcba240dfbf5fe7addced607e00c8931201af4050efd06243d54cf770f71530` |
| Actual finite configured CPU pass on both stacks | `/tmp/jax2onnx-erfc-denormal-replay-actual-v1/root-actual-dual-configured-pass.json` | `57ff851876f2575404729265737a7d4607b7fc61fc4be0ff92e633397726df38` |

The composition manifest at
`/tmp/jax2onnx-joint-main-integration-candidate-v2/manifest.json` binds the accepted
131/159/44/348 and original-six evidence. The final test-only successor is
`/tmp/jax2onnx-joint-stale-controls-candidate-v2/manifest.json`. Failed six-case
receipts remain under
`/tmp/jax2onnx-joint-MAIN-integration-v3/{primary-six,secondary-six}/run/receipt.json`;
final six-case proof is
`/tmp/jax2onnx-joint-MAIN-integration-v4/root-actual-dual-correction6-pass.json`.
Explicit pre-full typing reuse remains in
`/tmp/jax2onnx-joint-root-checks-v3/typing-result.json`. Earlier failed options,
matrix, retained-control and original-model epochs below keep their original
sources, criteria and result scopes.

## NNX Function state checkpoint

Accepted checkpoint, 2026-10-19: **`437d41fe`**, parent `b4a5a7ff`. Independent
review accepts the exact committed v10 sources and actual final integration
results. The isolated v9 adapter evidence also remains accepted on both supported
stacks: **13 new cases and 45 retained cases pass per stack**. The options
and matrix candidates were separate external work at this NNX-only checkpoint;
the later accepted union is recorded above. Public behavior remains LEGACY until the full B–F gates
pass across all existing functionality and fixed third-party sources.

The [shared Function state ports](../../../jax2onnx/plugins/_source_function_state.py)
and [NNX adapter](../../../jax2onnx/plugins/flax/nnx/_source_function_state.py)
carry real numeric owner state through named calls. Abstract evaluation, the
primitive implementation and delayed body tracing use the same pure state-call
descriptor. The implementation retains fixed topology, state types and actual
update/alias effects, and restores owner values, variable identities and aliases
on failure. Mutable owners must belong to the current trace; external preparation
uses the real NNX graph and state. Original source, decorators, defaults,
parameters and live controls remain unchanged. Forwarded-key wrapper handling
is a primitive-boundary alias proof, not a claim of original eager Python object
identity. General fixed-source preparation and whole-carrier JIT remain open.

The [new regression module](../../../tests/extra_tests/converter/test_source_nnx_function_state.py)
has ten functions / 13 cases, covering actual state feedback/reset, non-RNG
updates, multiple owners and aliases, native/source/legacy interleaving,
key-reuse rejection, local debug-context restoration and failure rollback.
Its three positive target cases produce **three artifacts, six CPU sessions and
48 runs per stack** under the local test assertions. The separate 45 retained
cases save 50 ONNX files and 96 target-array files per stack. Saved-file counts
corroborate local evidence; they are not process-wide runtime instrumentation.
The earlier adapter failures remain preserved. The final test correction removes
observer-induced key consumption and repairs iterator/capture handling while
retaining the existing model calls, negative checks and numerical criteria.

The preceding v9 complete primary suite passes 9,557 tests, retaining all
9,544 predecessor identities, but four typing errors keep its quality gate open.
V10 adds only one local NumPy Boolean-array annotation and three
`typing.cast` identity wrappers in two files. Independent normalized-AST review
accepts that type-only delta; the test, model, metadata and numerical criteria
are byte-identical to v9. Its external shadow mypy run passes for 545 files.
Those v9 runtime results retain their original provenance. On applied v10,
normal typing now passes for 545 files plus the RNG report, and pinned Ruff and
Black pass. A fresh secondary selection passes all 58 cases (13 new + 45
retained), with 174 phases and no failures, errors or skips. All 885 pinned
Python files, 57 extra inputs and the NNX inventory hold. The fresh v10 complete
primary suite passes **9,557 tests / 28,671 phases**, with four identical worker
collections and the same test-identity multiset as v9. No failures, errors or
skips occur. All seven installed normal commit hooks pass without source drift.
The local commit contains exactly the five reviewed source/test files and the
manual inventory; the working tree is clean. No push or publication is recorded.

The NNX-only inventory is **230 files, 1,500 functions and 748 migration
additions**, SHA-256
`828861f49a83aa74a188ffffb7d4cecc034095735c240848694f39c3aa8d2327`.
All 752 original manual anchors, all 1,490 predecessor function identities,
1,802 raw rows / 3,604 caller-x64 cells and all 63 opaque records are retained.
The separate normalization-options inventory is not transferred by this checkpoint.

| Final code-checkpoint gate | Actual independently checked result |
| --- | --- |
| Actual commit and parent | `437d41fefa0698532925916742452e7214991001`, parent `b4a5a7ff53f2e0689a490d2ff9fa57eb2669577c`; exact six paths, clean working tree |
| Complete primary suite | 9,557 pass = 9,544 retained + 13 new; 28,671 phases, four equal collections, no failures/errors/skips |
| Secondary 13 + 45 integration selection | Actual v10: 58 pass, 174 phases, no failures/errors/skips; this is not a full-secondary run |
| Applied typing, RNG and standalone quality | Actual v10: normal typing for 545 files plus RNG, pinned Ruff and Black pass |
| Seven normal commit hooks | All seven pass on the committed postimages; no bypass or fix drift |
| Exact committed-source independent closeout | ACCEPT; exact six postimages, 885 Python / 57 extra pins, test identities, actual terminal results and hooks verified |

<a id="original-models-now-reach-the-matrix-barrier"></a>

### Earlier original-model matrix barrier

The unchanged Functions014 ordinal 0/1 and Functions015 ordinal 0, each with
caller x64 off/on, completed **six barrier-exploration checks per stack**
at that NNX-only epoch.
They retain their original FP32 recipes, fixed B=5 or declared symbolic B=2/5,
one factory instance per cell, both live Dropout sites and the original
LayerNorm → Linear → exact GELU → Dropout → Linear → Dropout composition.
Caller x64-on is not evidence of an FP64 model.

Each stack records 20 native-reference groups over original eager, original
NNX JIT, prepared eager and prepared JIT, with T/F/F/T controls. Eight source
boundary observations identified the actual `dot_general` ordinary/equal-rank
matrix-profile exclusion. Every model record in that epoch remained
`OPEN_FIRST_MATRIX_BARRIER`: no model ONNX artifact, no target numerical
acceptance, and zero explicit ORT constructor/run guard attempts. The existing
live-Boolean preparation prerequisites remain recorded. The earlier first-cell
`TraceContextError` was retained history; the matrix gap then became observed
implementation work rather than a prediction or an ONNX impossibility.
At that epoch the matrix candidate was external. Independent static review
authorized 44 focused cases on an exact NNX-v10/options-v2/matrix snapshot. The later
actual run on each stack passes eight direct cases, fails the ninth and leaves
35 unexecuted. Eight artifacts, 16 CPU sessions and 36 target runs are recorded.
The first named NNX conditional fails during activated eager execution because
the custom SOURCE Function primitive has no JAX CPU MLIR lowering; that named
case reaches neither export nor target execution. This is the previously
recorded private whole-carrier execution gap, not an observed tensor-matrix
lowering failure or an ONNX impossibility. The later test-only successor checks
the actual exported body and ordinary execution after patch restoration, retaining exact original references and the
separate unresolved whole-carrier obligation. It now has finite matrix 44
acceptance, followed by retained 348 acceptance. The options/matrix postimages
are composed only through the exact reviewed union; none of these later results
relabels the first named-call failure. The latest original-model `erfc` barrier
and remaining limits are recorded in the [joint status](#joint-normalization-and-matrix-integration).

<a id="nondefault-layernorm-options-remain-open"></a>

### Earlier nondefault LayerNorm reference failure

At that epoch, external options-v2 had static acceptance only. Its actual run
collects 124 cases on each stack, executes 17, passes 16, fails one and leaves
107 unexecuted. The failed cell is Linen LayerNorm ordinal 3,
`raw-x64-off`, with its original default epsilon. Saved B=2/5 signed-zero/NaN
inputs give positive zero in native eager and disabled-optimization ORT but
negative zero in fresh JIT at the same recorded positions. Independent review
confirms contradictory native reference signs; it does not establish a lowering
limit or authorize a target waiver. The failed cell never reaches ORT's `all`
optimization setting, and that row's caller-x64-on target case remains
unexecuted in this failed options epoch.

The first seven new-option cell receipts and 16 passing prefix tests did not
accept that 124-case selection or its 18 remaining rows / 36 cells. Original
source/defaults/parameters, raw comparison criteria, target zero requirements
and strict typed-dataflow checks remain in force. No matrix, options or
complete-model acceptance is included in the NNX code commit.

### Completed native-only reference survey

The separate survey completes all **30 original rows × both caller-x64 settings
= 60 cells on each stack**, with unchanged source recipes, preparation and native
reference helpers. Each stack records **384 references in four roles = 1,536
helper-level model evaluations**, 120 completed tracing sites, 44 factory
instantiations and 16 cells retaining their original Equinox owners. The four
roles are original eager/fresh JIT and activated eager/fresh JIT. All source and
framework pins and import-origin checks hold; all export/ORT guards remain
unattempted. These counts describe the explicitly observed helper sites, not
process-wide instrumentation.

Independent inspection verifies every cell receipt, native/mask NPZ hash,
shape, dtype, mask and zero word on both stacks. All original/activated typed
traces match. The two stacks have the same 384 native NPZ payloads and exactly
**six conflicting configurations / 48 opposite-zero-sign positions per stack**:
Linen ordinal 3 and NNX ordinals 7 and 8, each under caller x64 off and on. Each
configuration has two positions in the B=2 signed-zero/NaN sample and six in B=5.
Every conflict is native eager `+0` versus fresh JIT `-0`, with FP32 outputs.
The other 54 configurations have no zero-sign conflicts. There are no dtype,
shape, NaN, signed-infinity, finite-mask or zero-versus-nonzero disagreements.
The 6,798 other finite-value differences per stack satisfy the unchanged dtype
bounds and available original raw tolerances in both directions. The survey
does not prove the compiler transformation responsible for the zero signs.

The reviewed test-only qualifier plan requires **exact eager zero words
everywhere**. Exact JIT zero words remain required outside the literal
pretarget-proven contradictory masks. Before export, each mask must match the
original row/contract, actual caller setting, sample label, input bytes, shape,
indices and both native words; unexpected, missing, reversed or nonzero
conflicts fail. Target values cannot select or broaden the mask, and a target
`-0` at a qualified point still fails the eager `+0` requirement. Existing raw
tolerances and all dtype/shape/NaN/infinity/finite checks remain unchanged for
both references. This resolves an impossible simultaneous native bit oracle
without weakening the target's exact eager-zero obligation. The native survey alone provided neither qualifier nor target acceptance.
The later options-v3 implementation and actual131-case selection now have
independent finite acceptance, as recorded in the
[joint status](#joint-normalization-and-matrix-integration); the original failed
options-v2 epoch remains unchanged.

| Evidence scope | Local receipt | SHA-256 |
| --- | --- | --- |
| Actual v9 adapter 13, both stacks | `/tmp/jax2onnx-nnx-function-state-isolated-v8/root-v9-actual-dual13-pass.json` | `5400ee8eb6408d647970f1455cf5409fee8d5d1819e50b70f8b436eab871449c` |
| Actual v9 retained 45, both stacks | `/tmp/jax2onnx-nnx-function-state-isolated-v8/retained45/root-v9-actual-dual45-pass.json` | `e6983c144dec9f7fcb98c8365d4b5d1a865a116a6bf51315f729484287e1e5ae` |
| Actual six-model barrier exploration, both stacks | `/tmp/jax2onnx-shared-b-model-isolated-v2/attempt2/root-actual-dual6-matrix-barrier.json` | `2272f00cde464b9165170d37fd1955ecddb15a1fb6d6c0070b68d488394e7f20` |
| Independent finite adapter / model-barrier acceptance | `/tmp/jax2onnx-nnx-function-state-astra-review-v9/actual-verification.json` | `48b4f5dbde118b5cd160ef160bc8a917c1c0a06801f258d8553feae7f1a1ce80` |
| V10 type-only delta review | `/tmp/jax2onnx-nnx-function-state-astra-review-v10.json` | `7bd6fc2eb4edb03e4d8386dc81d12c5b7e29e9bf4332a729525a7befd127b72c` |
| V9 complete primary, typing gate still open at that epoch | `/tmp/jax2onnx-nnx-function-state-MAIN-integration-v1/root-v9-full9557-pass.json` | `825771355ebba936b11d1768c7b20acaacbb004a7b39e80210928f4209486607` |
| Actual v10 normal quality and secondary 58 | `/tmp/jax2onnx-nnx-function-state-MAIN-integration-v2/root-v10-quality-and-secondary58-pass.json` | `14d9b486b3fb9f836d0dea6464c722fa5cdd5db8f44b5084abb2fb8d8d3b19ef` |
| Native zero-sign conflict; options acceptance open at that epoch | `/tmp/jax2onnx-normalization-options-zero-reference-astra-v1.json` | `0819a74e1dbb99cc55ef01473ca238b70fd501dcfad56777dd9ef372c16071e2` |
| Actual final v10 complete primary and secondary/quality evidence | `/tmp/jax2onnx-nnx-function-state-MAIN-integration-v2/root-v10-full9557-quality-secondary58-pass.json` | `e81206757e0fa77e44b88ccaf98641934da1ec93300e254884a7eb454559af5a` |
| Actual NNX-only code commit and seven normal hooks | `/tmp/jax2onnx-nnx-function-state-MAIN-integration-v2/root-v10-checkpoint-committed.json` | `c12991adb657211cd1956b12699a33067117a6e369941a44e03499f47e79e2b7` |
| Independent exact committed-source acceptance | `/tmp/jax2onnx-nnx-function-state-astra-committed-review-v10.json` | `54e02cffd4e675631f08bf8919db6965f68c0bb65d0876f47f29a293830138fb` |
| Independent complete dual-stack native survey acceptance | `/tmp/jax2onnx-layernorm-native-reference-astra-actual-review-v1.json` | `e7af7be4aefa6f9804b09e6ef0b1a9a1469b81b4b6979f177eb115d8b5bf1e74` |
| Complete native reference/mask/word and criterion audit | `/tmp/jax2onnx-layernorm-native-reference-astra-data-audit-v1.json` | `f5850ce870a75fecbbc4e684b201b0e3a1bd5a1a181012f5235cb02a19ec670c` |
| Bounded pretarget native-zero qualifier plan; implementation open at that epoch | `/tmp/jax2onnx-layernorm-zero-oracle-astra-plan-v1.md` | `15f7a1b4ce58552cd08b373eede82f22a38404c7bb973e30d1d1e32b54310aa8` |
| Matrix static acceptance for finite isolated runtime; no runtime result | `/tmp/jax2onnx-source-tensor-matrix-astra-review-v1.json` | `28f112df624fb864d8367903765419925abf612bc1f5d85d732136602ccc17c6` |
| Later actual matrix prefix: eight pass, named activated call fails; 35 unexecuted | `/tmp/jax2onnx-source-tensor-matrix-isolated-v1/root-actual-dual-matrix44-first-failure.json` | `8c56093dfb45f23b2b27fb754040f5ed286f988349f5cee6d2ba8374b6dc6442` |

Local evidence paths are provenance, not tracked code dependencies. The committed
NNX result, completed native survey, failed options/matrix epochs and later
accepted finite successors retain their separate scopes.

<a id="shared-batch-normalization-work-in-progress"></a>

## Shared-batch normalization checkpoint

Outcome at this preceding checkpoint, 2026-10-19: **`bb7a7d76` is independently
accepted**. The complete primary suite passes **9,544 tests**, the repaired
focused selection passes **172 tests on each stack**, and all seven actual
commit hooks pass. The independently accepted **124-case finite selection on
each stack** retains its original scope; the earlier corrected core selection
passes 526 tests per stack. Standalone quality checks pass and the reviewed
manual inventory is committed. The failed 65-case and 114-case family epochs,
and the first broader integration runs with twelve failures per stack, remain
recorded below. In particular, the earlier 4,876-case secondary run remains a
failed run; the new secondary selection covers 172 cases. This checkpoint does
not establish complete family or model acceptance. Public behavior remains
LEGACY until all B–F migration gates pass.

The corrected core admits one atomic leading B at ranks one through four
through both export and sample validation. Declared NHWC/NCHW boundaries retain
the exact fixed permutation, common B, logical dtype and actual artifact ports.
Matrix and PRNG handlers retain their narrower operation-specific shape rules.
The core terminals are primary `60101/381fe3` and secondary `13534/146291`,
both exit 0. These focused passes do not establish family numerical acceptance
or bind later repair bytes.

The first 65-case family selection remains failed: **18 passed, 32 failed and
15 fixture errors per stack**. Its terminals are primary `86019/cd04e5` and
secondary `16604/10f3da`, both exit 1. Ten original LayerNorm cells per stack complete
their new checks, with **20 CPU sessions and 122 target runs per stack**.
Those successful cells are partial sample evidence within a failed selection.
The historical core and family receipts both inherited a stale
`runtime_started: false` field. Their recorded terminals and counts are valid;
the separate execution clarification confirms that both selections actually
ran on both stacks. The original receipts remain unchanged.

The first failure review identified a real missing capability in original NNX
LayerNorm row 13: its captured strong FP32 scale and bias reshape from `(576,)`
to `(1,3,3,64)`. Singleton-only static admission could not preserve that unchanged
model. The implemented amendment permits ordinary unpermuted floating reshapes
with positive fully static extents, equal exact Python-integer element counts,
exact native sizes/result, unchanged dtype/weakness and no placement. This
applies to runtime data as well as constants. Symbolic reshapes retain the
unchanged leading B and singleton-only restriction. The existing Reshape
lowerer supplies the representation; the amendment does not require a different
formula or model. The second family run exercises this correction without
establishing complete family acceptance.

The initial oracle also rejected valid representations: internal INT64 shape
helpers without declared `ValueInfo` types, and a harmless unused import of a
Function's own domain. The bounded correction proves the actual shape
def-use chain and any existing type/shape declarations, and retains exact
resolution and ports for every used Function call. Graph boundaries, floating
arithmetic and unexplained domains receive no blanket exemption.

The second family terminals are primary `18394/b6b6b3` and secondary
`75755/4ac80d`, both exit 1: **110 passed and four failed out of 114 cases per
stack**, with no fixture errors. Twenty original LayerNorm
cells per stack complete their checks: **40 CPU sessions and 262 target runs
per stack**, still partial evidence in a failed selection. The four remaining
failures on each stack are original Equinox rows 4–7 at raw-x64-off. Saved
artifacts show that all graph nodes are reachable from outputs, but one unused
FP32 scalar NaN initializer remains. Its live singleton counterpart has the
same bytes. The raw-x64-on graphs have no unused initializer.

Installed-API and converter-source inspection explain the retained initializer:
the removed Cast nodes keep stale input uses when `Graph.remove` uses its
default unsafe removal. The original in-memory use lists were not saved.
The narrow repair adds `safe=True` to exactly three existing removal calls,
after their existing rewiring. It preserves cast-fold eligibility, arithmetic,
observed intermediate casts and the strict consumed-dataflow audit. Independent
static review accepts those changes and ten public-IR regression cells across
SOURCE and LEGACY. Applied-source and selector readiness are also independently
accepted. The repaired focused runs finish at primary `53412/6934af` and
secondary `23391/3ddd85`, both exit 0. Independent review accepts these actual
finite results. It recomputes 468 saved target runs on each stack, including
all 24 original caller cells and all six original opset-16 cells, 18 consumer
cases and 18 static reshape cases. All eight repaired Equinox artifacts per
stack contain no unused initializer and retain the live NaN bytes `0000c07f`.
Eight additional typed-boundary runs are supported by the passing frozen test,
without separately saved output arrays. Actual Function/If and compiled-body
proofs retain their finite scope; whole-carrier JIT remains unestablished.
Earlier raw structural callback failures remain in
the failed-epoch records; they do not describe the repaired run's outcome.

Before the four-file repair, standalone checks pass: global Ruff (`951d49`), Black checking
of all 23 changed Python files (`8e52ba`), and configured typing for 543 files
plus the RNG trace check (`29480/9107f3`). The strict public MkDocs build passes
at `6437/213594`, exit 0, including the updated optimizer guide. These eight
internal precision documents are outside that build. After the four-file repair,
global Ruff and pinned Black checking of all 26 changed Python files pass.
Configured typing also passes for 543 files plus RNG (`3431/97ec8a`, exit 0),
with the source freeze unchanged. All seven installed commit hooks later pass
on the exact committed code, as recorded below.

The independently reviewed manual inventory at this checkpoint is **229 files,
1,490 functions and 738 migration additions**, SHA-256
`5603f7a37bcbd461dfc9f60d65bdd202afc253ee43a909438cc1acccacd4a05d`.
All 752 original anchors and all 1,460 preceding function identities are
retained. Raw rows, x64 cells, opaque records and all nonmanual fields remain
unchanged. The full-primary run (`16806/2005a5`) and 63-module secondary
affected-area run (`79239/6ef7b9`) both finish with exit 1 under the unchanged
freeze of 882 Python files, 56 extra inputs and the inventory. All preceding
9,296 primary test identities are retained. The same twelve checks fail on
both stacks; all setup, call and teardown phases occur once per case.

Nine failures rejected the fixed audited Add/dot control set. A bounded follow-up
created its four artifacts on each stack without ORT execution: schema,
source audit, artifact digests and structure are valid, and runtime/numerics
remain unverified. The dot counterpart fingerprint includes the entire oracle
file, which changed with the normalization extension. Its reviewed replacement
fingerprint is `804e40007771d93d2c4e2f2fbd6746c0d79c0eaa2755dca212e4127a6e4ca1c7`;
the fixed control still expected the preceding fingerprint in the failed run.
The repair retains a fixed literal rather than a self-derived expected value,
preserving the corruption guard. Original raw dot callbacks remain
failed, as required by these controls.

Three further checks had stale expectations: the counterpart-type error now
lists `NormalizationCounterpart`, a symbolic batch reduction has its specific
rejection message, and symbolic Max is now admitted while symbolic Min remains
excluded. The repaired tests preserve the negative guarantees and acknowledge
the reviewed Max extension. Sol implementation and Astra review were briefly
interrupted by model usage limits; availability was restored on retry. The
four-file repair has independent static acceptance and was applied by ROOT
after exact preimage checks (`538af9`, exit 0). It retains the independently
reconciled dot fingerprint as a fixed literal,
updates the two diagnostic expectations, and pairs the admitted symbolic-Max
case with the retained symbolic-Min rejection. Production admission and
numerical criteria are unchanged.

The four-module, **172-case focused selections pass** at primary
`97112/c9a6f1` and secondary `68631/794baf`, both exit 0. The freeze of 882
Python files and 57 extra inputs, including the v3 documents, holds. Independent
review accepts the actual finite results and the full-run prerequisite. The
complete **9,544-case primary suite passes** at `64293/9cd29a`, exit 0, with
zero failures, errors or skips. All prior 9,544 identities are retained; all
28,632 setup/call/teardown phases occur exactly once across four workers.
The 882-file source snapshot, 57 extra inputs and inventory remain unchanged.
The new secondary 172-case selection is narrower than the earlier failed
4,876-case affected-area run; it does not turn that historical run into a pass.

Independent integration review accepts the result. The normal local code
commit finishes at `62913/f31e25`, exit 0: all seven installed hooks pass,
and commit `bb7a7d765c519aba0ba73437af59e681aef585ad` has parent
`651ce09e35c454f2856aa31b073b191bb146fa8c`. Independent closeout verifies all
28 committed blobs, the source/extra-input/inventory freeze, the restored eight
v3 documentation files and empty index. This closes the code checkpoint;
the final documentation update is separate. No model candidate or ONNX artifact
is committed, and no public transition follows.

All **12 original default LayerNorm rows / 24 raw x64 cells** have finite acceptance,
including NNX row 13 and all six original opset-16 cells. The remaining
18 nondefault-option rows / 36 cells were explicit implementation work at that
checkpoint; their later finite acceptance is recorded above. Original
model recipes and numerical criteria, the 1,802 raw rows / 3,604 cells and all
63 opaque records remain unchanged. At this checkpoint, full Functions014/015
follow-through was next. Its external test candidate ran only in an isolated
worktree, using the then-current 882-file source snapshot plus two accepted
candidate files. Both exploratory runs failed their first case: six collected,
one executed and five unexecuted per stack, with no retry, completed cell
receipt, returned ONNX artifact or ORT execution. First-cell original/prepared
eager/JIT references completed the zero/nonzero T/F/F/T sequences before the
facade failed; this did not establish six-cell coverage.

The first observed blocker was Flax `TraceContextError`: `FunctionPlugin` called
`jax.eval_shape` and the original nested MLP's `dropout1` attempted to mutate
`RngCount` from another trace level. The generic Function/NNX adapter was then
external work, and neither the predicted matrix gap nor erfc had been reached.
Imports were guarded to that worktree, MAIN remained unchanged and no model
tests were transferred. This failed exploration remains preserved; the later
[NNX checkpoint](#nnx-function-state-checkpoint) records the accepted adapter and
six-cell matrix-barrier observations without turning them into model acceptance.

| Evidence scope | Local receipt | SHA-256 |
| --- | --- | --- |
| Corrected core pass, both stacks | `/tmp/jax2onnx-shared-b-focused-v2/root-dual-core-pass.json` | `92792f56eea5af07d6a46fbebd146ba3fd2c1e84972d8104b725e4751646d998` |
| First family failure, both stacks | `/tmp/jax2onnx-shared-b-family-runtime-v1/root-dual-family-failed-v1.json` | `0fe4b9676b2ead0630854c5116f0713b6b8b59a135fd2947acebb33758aad50f` |
| Second family failure, both stacks | `/tmp/jax2onnx-shared-b-family-runtime-v2/root-terminal-failed-v2.json` | `9ea1851db42a7e4bf4e0106fefe88159bcf28b47c0b076f727edec0eaf61e41f` |
| Actual-execution clarification | `/tmp/jax2onnx-shared-b-focused-v2/root-execution-clarification.json` | `9c3239d3032e39919c1ccd441c1c4cdbbcf4f233ba1d6612af5a26fb303019e3` |
| Bounded repair amendment | `/tmp/jax2onnx-shared-b-family-failure-astra-amendment-v1/amendment.md` | `94a314368afdad42c6748f5b504bf3c6ae4f98b5a8f04e9aaf453de843b727da` |
| Initial amendment review; scope approval only | `/tmp/jax2onnx-shared-b-family-failure-astra-amendment-v1/verification.json` | `ba7b347e02f1e2e7047f69c158e3e445d4cf14f01c812924332c6960fd6d9d65` |
| Three-call Cast repair and ten regression cases | `/tmp/jax2onnx-shared-b-cast-removal-repair-v2/manifest.json` | `0b04cba716e5e3be9e7e21c547e734dcf26cb8e0f84dfa0189bf64f671465417` |
| Cast-repair static prerequisite review | `/tmp/jax2onnx-shared-b-cast-removal-astra-review-v1/verification.json` | `f0ce1486cf7a64d5ce840ef09059ae76f5e0d8d1a141403ff171a45ebbbd3b30` |
| Applied-source and selector readiness only | `/tmp/jax2onnx-shared-b-cast-removal-applied-astra-review-v1/verification.json` | `00c34953724c4d5304970c1f8194a6d098c5286a9dca99dd9bbfdd4391b073a8` |
| Repaired focused pass, both stacks | `/tmp/jax2onnx-shared-b-family-runtime-v3/root-dual-focused-pass.json` | `0ab07085586472de6401d63ce76d96b896f4667cdee8f2bff9267a0eaf2ab08e` |
| Independent actual finite acceptance | `/tmp/jax2onnx-shared-b-finite-actual-astra-review-v1/verification.json` | `6b4187432ce804920294c0627a4e58f5e3568e582bc39d1bbd491a688d45014d` |
| Independent manual-inventory reconciliation | `/tmp/jax2onnx-shared-b-inventory-sol-verification-v1/verification.json` | `6e8c3126cfbed1f28c41dc4219279d76163207408c63392c141007d19e9a9799` |
| Failed full-primary and secondary affected-area integration | `/tmp/jax2onnx-shared-b-MAIN-integration-v1/root-dual-integration-failed.json` | `cd968682e608c3311679d323853f78b4d761f2130dd01f43a332913e5dca80a2` |
| Four-file repair and v3 documentation static acceptance | `/tmp/jax2onnx-shared-b-integration-repair-astra-review-v1/verification.json` | `7511842ea736c5034ee7d66340276d59cc65ad904485716bf5ab070fbf390815` |
| Actual repair checks pass on both stacks; full primary released | `/tmp/jax2onnx-shared-b-MAIN-integration-v2/root-dual172-pass-and-full-release.json` | `c5cdf27e9d641b71d476cbc2e5beaf7078404971b30a2c97f4565ac390a89f58` |
| Independent actual repair-result acceptance | `/tmp/jax2onnx-shared-b-integration-dual172-astra-review-v1/verification.json` | `eefd7db4366bb584b077d7bfd855e85939686cfaa06bea3c082e554daa3a9fc9` |
| Complete primary and standalone quality pass | `/tmp/jax2onnx-shared-b-MAIN-integration-v2/root-integrated-checks-pass.json` | `b81a46494f0b5ec691de7c29f5974ef615d71bad61b2e686a8ae3faa1d9c173f` |
| Actual code commit and seven installed hooks | `/tmp/jax2onnx-shared-b-MAIN-integration-v2/root-code-commit-verified.json` | `4d882b1e8f855dfb54a5464aec440c4d25fdd668e11d06fb244f150fbd512bc1` |
| Independent actual code-checkpoint acceptance | `/tmp/jax2onnx-shared-b-code-commit-astra-closeout-v1/verification.json` | `e8b105efb822322939c5b71de5d861151f057411a370912c9e4efae12400adf5` |
| Separate isolated original-model exploration; both first cases failed | `/tmp/jax2onnx-shared-b-model-isolated-v1/root-exploration-failed.json` | `bfe7de0699cd10d7c471a27e233f50698d57eb51b2a192dda115605a0fe206d9` |

These paths are local review evidence; tracked code and tests do not depend on
them. Earlier accepted results and failed epochs below retain their original
scope and status.

## MLP state and bounded normalization acceptance

The 2026-10-19 checkpoint has independent acceptance for the finite original
NNX/Linen MLP runtime matrix and selected static normalization examples. Public
behavior remains LEGACY. These results do not complete the existing-functionality
inventory or authorize the final public transition.

The original ordinal-2 MLPs retain seed 17, dimensions 30 → 20 → 10,
running-stat BatchNorm, positive Dropout rate 0.1, the original GELU choice,
parameters and factory dtypes. External pure whole-model preparation exposes
actual state/RNG inputs and outputs while preserving both original branches.
Four independent reference roles are captured before SOURCE activation/export:
original eager, original NNX JIT with its static flag, prepared eager and
prepared JIT. Linen retains its original bridge and key contract.

The finite target matrix covers x64-off FP32, x64-on FP32 and x64-on FP64,
original zero and meaningful nonzero inputs, and B=2/5 where declared. Each
serialized artifact is reused with `ORT_DISABLE_ALL` and `ORT_ENABLE_ALL` on
CPU, on both supported stacks. T/F/F/T controls feed actual returned state into
the next call, with resets and replay. Every production output is compared
with each of the four native references using the original preapproved absolute
bounds, `rtol=0` and no NaN equivalence. All values are finite and retain exact
dtype/shape; zero outputs have exact positive-zero bits. UINT32 state words and
the separately observed Dropout masks match the original native values exactly.
The accepted Phase B packet contains 12 production/diagnostic model pairs and
40 selected PRNG artifacts across both stacks: 128 sessions and 2,144 calls.
Diagnostic Y has an ABI-only contract; direct-mask acceptance does not confer
numerical acceptance on that diagnostic output.

The tracked [permanent runtime regression](../../../tests/extra_tests/converter/test_source_mlp_runtime_acceptance.py)
passes its six NNX/Linen × x64/dtype cases on each stack. Each actual run has six
exports, twelve CPU sessions and 192 target calls, using B=2 and the same four
reference roles, state-feedback/reset/replay checks and frozen criteria. This
minimal CI test does not repeat the separate direct-mask matrix. Source,
configuration, PRNG, callable, policy and observer restoration checks pass for
the recorded successful runs; neither finite packet is a universal PRNG or
model-support guarantee.

The [variance/std checks](../../../tests/extra_tests/converter/test_source_square_variance.py)
and [LayerNorm checks](../../../tests/extra_tests/converter/test_source_layer_norm_exports.py)
add finite static FP16/FP32/FP64 evidence for native square, variance/std and
NNX/Linen/Equinox LayerNorm. They check actual serialized dataflow through
mean/variance, epsilon, square root, reciprocal, affine and residual operations,
including selected inline, named Function and conditional boundaries. Fourteen
valid-ONNX wiring mutations are rejected. The named-boundary proof evaluates the
same captured exported `ClosedJaxpr` eagerly and with fresh JIT against its
preactivation native references, including both conditional inputs.

That proof establishes **actual exported-body eager/JIT equality**. Whole-carrier
JIT equality remains unestablished; the named residual conditional's activated
eager call also triggers compilation and retains its missing-MLIR
`NOT_ESTABLISHED` result. The original failed epoch exits 1 with zero target
sessions/runs. Outer serialized wiring and full-model target comparisons for
both selectors have separate acceptance. Normalized zero rows use their existing
exact signed-zero comparator; nonzero bounds remain unchanged. The two actual
85-case runs are v4 executions (116 sessions/170 calls per stack); v5 acceptance
adds only the independently verified annotation-equivalent change. These results
do not establish complete normalization-family or Functions014/015 acceptance.

The earlier local integration brought 28 reviewed source/test paths and the
manual inventory update into MAIN. Its independent source/inventory review
verified the 880-file source freeze and all prior inventory anchors. The first
joined focused runs recorded 1,027 passes and the same one stale expectation
failure on each stack: the preparation test still expected profile admission
failure after the bounded implementation reached runtime checking. That failed
epoch remains in the record. The narrow companion repair has independent source
acceptance. The later full failure and repairs are recorded separately below;
the earlier review does not certify subsequently changed source bytes.

Frozen acceptance receipts are retained under `/tmp` as local review evidence;
the tracked tests have no dependency on those paths:

| Accepted scope | Independent receipt |
| --- | --- |
| Finite MLP/selected-PRNG target matrix | `/tmp/jax2onnx-mlp-phaseB-actual-result-astra-review-v1/verification.json` |
| Permanent MLP runtime regression | `/tmp/jax2onnx-mlp-permanent-ci-astra-runtime-review-v1/verification.json` |
| Static normalization and actual-body proof | `/tmp/jax2onnx-normalization-candidate-v4-astra-review-v1/verification.json` |
| Local MAIN source and inventory only | `/tmp/jax2onnx-mlp-normalization-MAIN-astra-review-v1/verification.json` |

Earlier sections below retain their original checkpoint evidence. Historical
barriers and failed criteria are not retroactively changed by later acceptance.

## Full-suite failure and finite repair status

The first full primary integration run is **FAILED**: 9,191 passed and 33 failed
out of 9,224 collected tests, with no skips or errors. All 27,672 setup/call/
teardown phases ran. Its actual terminal is session `73239`, chunk `845246`,
exit 1. The follow-up classification records eight stale expectations, three
non-Dropout product defects and twenty-two Dropout control failures. The frozen
failure receipt and source postimages remain historical evidence; later finite
passes do not change this run's result.

### Accepted finite non-Dropout repair

The actual v6 packet passes the same 57 cases and all 171 phases on each
supported stack. Each run records eleven successful exports, five expected
structured exclusions, twenty-four CPU sessions, sixty-eight target calls and
twenty-four validation reports. Actual terminals are primary `59789/c41a76`
and secondary `8795/2cb28f`, both exit 0. Independent review accepts this finite
packet. Its offline audit checks all 136 saved calls and 296 output leaves
against the original criteria: all observed leaves are byte-identical to their
references, including the thirty-two leaves whose approved comparisons remain
tolerant. No tolerance was changed.

The repairs cover the exact native uniform `0`/`1` literal proof for weak FP64,
the selected static rank-one PRNG width of 32, named initializer handling,
malformed-reshape diagnostic classification, and captured typed keys in actual
Functions. Actual key values remain logical source values; selected valid-key
mathematics is unchanged. Unsupported `rbg` keys retain a structured nested
profile exclusion. The rank-one width restriction is a private migration gap
outside this finite implementation, not an ONNX limitation.

Captured-key Function/If artifacts retain the actual key constant and required
nested dataflow. Source/settings/PRNG/policy/callable/observer restoration checks
pass for these actual runs. The earlier failed v4 and v5 epochs remain in the
closed packet, including the raw `ValueError` preceding the structured-error
repair. A subsequent path-header comment is AST-equivalent but was not executed
by v6. Finite diagnostic numerical validation does not establish general product
admission, complete PRNG support or whole-checkpoint acceptance.

### Explicit Dropout state preparation: first runtime failure

The new internal [state preparation helper](../../../tests/extra_tests/converter/source_state_preparation.py)
and expanded [Dropout checks](../../../tests/extra_tests/converter/test_source_dropout_parameters.py)
define an explicit adapter for the declared NNX and Linen Dropout owners. The
helper retains the actual original target code, body, defaults and nonowner
captures. Only the declared NNX owner closure cell is rebound to an owner
reconstructed from its real graph, fixed state, key and counter. Linen retains
its original fixed external key, rather than replacing it with a new state input.

The prepared workflow threads actual returned state through Function, If and
Loop calls. Its independent references use sequential calls to the original
native step, and the unchanged whole original eager/JIT model wherever valid.
Raw positive-rate NNX training with captured mutable state in If/Loop is natively
invalid and remains a separate prerequisite failure. For those cases, sequential
original-step and prepared eager/JIT comparisons supply the later finite evidence;
whole-original native execution is not established. Equinox retains diagnostic
sample-property criteria, without an exact JAX stochastic-mask claim.

The combined 53-case contract plans 67 exports, 134 CPU sessions and 2,308
target calls per stack. The first actual primary run is **FAILED**: 53 cases
were collected, seven executed, six passed and one failed; forty-six remained
unexecuted. All twenty-one phases of those seven cases ran. The actual terminal
is session `17295`, chunk `019246`, exit 1; pytest reported 95.95 seconds.

The original positive case
`test_dropout_uses_real_named_operand_through_nested_scope[positive-rate-if-nnx]`
failed while constructing its first `ORT_DISABLE_ALL` session at x64-off.
ORT reported nested If/Function `TypeInferenceError` at
`_inlfunc__dropout_state_named_node_If_2`, output `cond_then_0/cond_capture_2`:
inferred rank 0 versus declared rank 1. That epoch exposed a representation/
capture defect. The subsequent source-reviewed repair then needed runtime
validation; its later accepted result is recorded below. The original failure
does not establish an ONNX limitation.

The failed run retained thirteen returned exports, twenty-five session
construction attempts (twenty-four returned and one raised) and 384 actual target
calls. No secondary run or rerun followed that epoch, and no source edit was
made as part of closing it. The original numerical comparisons before failure
have not received independent acceptance. The planned full-matrix counts were
not reached, and that failed epoch supplies no runtime acceptance for the
combined packet.

The failure record confirms unchanged 881 guarded Python files, 29 Markdown
files, generated artifacts, dependencies and inventory. Settings, PRNG, policy,
watched callables, original/rebound target captures and the original preparation
body were restored. The observer itself was restored, but its separate
`all_actual_patch_targets_restored` flag is **false**. At that epoch the
original observer did not retain individual differences for its 590 targets;
the later diagnostic below provides separate evidence.
The aggregate `all_sessions_CPU_threads1` flag is also **false** and includes
the failed constructor. That epoch did not resolve its cause; this flag alone
does not establish incorrect thread settings. No all-restoration or all-sessions
acceptance is claimed for that failed epoch.

The helper remains an explicit finite preparation mechanism, not a generic
automatic adapter for arbitrary opaque fixed sources. Supporting unchanged
third-party JAX sources through honest explicit preparation remains part of the
full completion contract. The first failed epoch must remain visible beside
any later repair and successful execution.

### Source-reviewed naming and restoration repairs

A separate actual primary restoration diagnostic executes one case and all
three phases successfully, with two exports, four sessions and sixty-four target
calls. Its outer terminal is nevertheless **FAILED**, session `76426`, chunk
`477293`, exit 1: inherited attributes were restored as new local shadows.
The original record identifies two resolved-identity mismatches in DinoRoPE
classes; the read-only addendum identifies five namespace/descriptor leaks in
total, including three whose resolved identity appeared unchanged. This is a
real restoration defect. The diagnostic's passing pytest case does not establish
numerical or integration acceptance, and its failed outer result is retained.

Naming v2 and restoration v1 now have independent **source-component acceptance**;
their production postimages are retained in local MAIN, with the later reviewed
two-line test-only public-call correction described below. Naming assigns collision-free
local namespaces while preserving external Values and names, and detaches the
shared proto-backed tensor storage whose name setter would otherwise mutate
another owner. Nested owned definitions retain distinct names without changing
stable formals or free captures. The rejected naming v1 and its occupied-name/
shared-storage blockers remain part of the review history.

Restoration now snapshots raw direct class/module attributes, removes introduced
inherited shadows and unwinds leaf patch contexts in reverse order. Factories
still receive resolved originals; the protected activation block also covers
entry failures. Existing instance/slot behavior and Function reference counting
remain intact. The source review used static and standard-library stand-in
checks; it did not execute the proposed real IR or numerical regressions.

The joined 101-case packet combines the original 53 cases with twelve naming
and thirty-six restoration cases. Its first two attempts each record 35 passes,
one failure and 65 unexecuted cases (108 phases). The v3 attempt passes a private
`precision_policy` keyword to public `to_onnx`, which rejects it before any
export/session/run. An independently reviewed two-line test-only deletion
restores the ordinary public call; it changes no production API.

The v4 attempt reaches one delegated serialization and saves its protobuf,
but returns no UI export and opens no session or target call. The external
observer incorrectly requires the entire public IR to remain unchanged during
serialization and records a mismatched inherited-child baseline first observed
while its ancestor patch was active. Its saved BEFORE includes an unnamed
initializer tensor; the installed serializer assigns initializer tensor names.
No actual AFTER snapshot was saved in that failed epoch, so the complete delta
is unestablished. It cannot be retrospectively certified as a name-only change.
These test/observer failures establish neither a new production defect nor an
ONNX limitation. Both original failed closures remain unchanged.

The amended external observer now records actual BEFORE/AFTER transitions,
separate delegate/wrapper outcomes and proven inherited-target baseline lineage.
Its exact v5 prerequisites have independent acceptance. Two installed-IR-only
toy serializations on each stack provide separate observer-mechanics evidence;
they neither reconstruct the missing AFTER nor establish model results.
The subsequent actual dual-stack result and its separate independent acceptance
are recorded next.

### Accepted dual-stack finite runtime result

Independent actual-result review accepts the exact joined101 packet on both
supported stacks. Each completes **101 cases / 303 phases**, with 73 returned
exports (72 protobuf and one IR), 74 serializations, 141 CPU sessions, 2,364
target runs, 18 samples, six validations, eight LEGACY comparisons and 507
original target-factory constructions. The primary terminal is `30813/10bda9`,
secondary `61217/6ec45d`, both exit 0. These actual counts supersede planned
counts for this successful packet; they do not relabel any earlier failure.

The independent audit checks every saved target run against its original
criterion, including 4,096 exact NNX/Linen target-to-original output comparisons
per stack. It preserves original sequential-step, prepared eager/fresh-JIT and
valid whole-body references, T/F/F/T controls, reset/replay and returned UINT32
key/counter chains through Function, If and Loop. Linen's fixed original key
remains `[0,31]`. Equinox training retains its original zero/scaling property
criterion; exact native stochastic-mask equality is not claimed. NNX whole-body
JIT uses the owner-aware route, and raw captured mutable training If/Loop remains
an invalid native prerequisite. The explicit state helper is still finite,
without a generic adapter for arbitrary fixed sources.

All 74 actual serialization transitions and 141 model-blob lineages per stack
pass the frozen contract. The installed serializer synchronizes initializer
`Tensor.name` with `Value.name`; 1,128 permitted name assignments per stack are
witnessed. Returned IR is accepted through its actual same-model transition
chain, without claiming raw IR metadata immutability. Snapshots cover selected
identity, wiring and metadata fields, not arbitrary payload/metadata equality.
The missing actual AFTER snapshot from the earlier v4 failure stays missing.

Restoration passes for 638 canonical patch targets across 44,023 saved records
per stack, using exact local ownership, descriptors, resolved binding and
recorded ancestor provenance. A stricter supplementary snapshot comparison
initially failed on one marshaled-code digest. That diagnostic FAIL remains
recorded: the same function/code identities and every frozen restoration field
match, and stdlib controls on both exact Python runtimes reproduce unstable
marshal digests for unchanged code objects. Actual target marshal byte streams
were not saved, so their precise byte-level cause is unestablished. No frozen
restoration criterion was relaxed. Full current lookup-lineage arrays were not
saved; the audit reconstructs expected lineage from first-seen/ancestor records,
checks current targeted ancestors, and retains the original runtime assertions
for untouched ancestor namespaces.

The eight LEGACY reference arrays were computed after the target runs; their
original tolerant comparisons pass. The named subtraction test retains its
passing separate JIT assertion, while only its eager reference was saved for
independent array review. These limits stay separate from the exact state and
output evidence above. The dual-run closure retains its original pending-review
label; the separate `verification.json` records the later independent **ACCEPT**.
This finite result does not complete all model, PRNG, normalization or deployment
coverage.

### Integration and inventory gates

After the five naming/restoration postimages, actual standalone configured
typing passes for 543 production files plus the RNG trace check. Global Ruff
and pinned Black checking of all 41 changed Python paths also pass. These v4
results apply to the current repaired source; earlier v3 check receipts retain
their original scope. Standalone checks remain separate from actual final hooks.
The fresh primary full run now passes **9,296 tests and all 27,888 phases**
across four workers, with zero failures, errors, skips or omitted cases. It
retains every identity from the earlier 9,036-test baseline, 9,224-test failed
full run and original 1,028 focused cases. Its actual terminal is `58291/4947b5`,
exit 0. The secondary affected-area run passes **1,052 tests and all 3,156
phases**, including every original focused identity, at `27310/5f8aed`, exit 0.
Independent review accepts those actual integration gates and their exact
source/inventory binding. It does not claim a full secondary-stack suite.

The primary audit binds source and test identities; the secondary additionally
checks its recorded settings, PRNG, policy and watched-callable restoration.
All 881 Python, 29 protected Markdown, 26 generated and three dependency files
match the accepted freeze. Earlier failed epochs remain failed history.
The separate actual configured-hook run passes all seven hooks with no source,
documentation, inventory, dependency or staged-path changes (`31329/e7942a`,
exit 0). The subsequent normal local code commit
`7d74a2001ef0a8e20a08151e7d67e20b1ae64150`, parent
`d960d2c50c39da275af364ddadadc7f98d18b1b1`, contains exactly the 42 reviewed
paths and passes all seven actual commit-time hooks without skips or bypass.
Its terminal is `72344/418bb2`, exit 0. Verification confirms matching tested
bytes, unchanged 881 source files, 29 precision Markdown files, 26 generated
files, three dependency files and inventory, with a clean worktree. The code
commit establishes a local checkpoint; no push, release or public transition is
claimed. The strict public MkDocs build also passes on the accepted code with
its 45 public documentation/configuration inputs unchanged. These eight internal
precision documents are outside that build; their content/link review, application
and separate documentation commit remain subsequent gates.

The reviewed inventory is now applied: **228 manual files, 1,460 functions and
708 migration additions**, exactly candidate `bbecd123b81855d694e42e59be2f18126c6fe88e0f1a0bb044ddabfc668fdc53`.
Actual integration review verifies every preceding 1,440 and intermediate 1,448
function identity plus all 752 original anchors, with zero removals. Only manual
rows/provenance and their function count change; all 1,802 raw rows/3,604 cells
and nonmanual fields remain exact. Applied inventory and integration acceptance
and the separately accepted finite runtime result leave the full migration
gates open.

| Evidence scope | Local receipt |
| --- | --- |
| Failed full primary epoch | `/tmp/jax2onnx-mlp-normalization-MAIN-integration-v2/full-primary-failed-verified-v1.json` |
| Accepted actual finite non-Dropout v6 | `/tmp/jax2onnx-full-v2-non-dropout-actual-astra-review-v1/verification.json` |
| Closed dual packet, including v4/v5 failures | `/tmp/jax2onnx-full-v2-bounded-repairs-v1/non-dropout-dual-v6-closed.json` |
| Original combined Dropout contract | `/tmp/jax2onnx-dropout-state-preparation-runtime-v1/closure.json` |
| Failed first combined Dropout primary epoch | `/tmp/jax2onnx-dropout-state-preparation-runtime-v1/combined53-primary-v1-terminal-failed.json` |
| Failed restoration diagnostic and five-leak addendum | `/tmp/jax2onnx-dropout-observer-candidate-v3/restoration-probe1-primary-v1-terminal-failed.json`, `restoration-probe1-five-namespace-leaks-addendum.json` |
| Naming/restoration source components only | `/tmp/jax2onnx-scope-restoration-source-astra-review-v1/verification.json` |
| Current standalone typing, Ruff and Black | `/tmp/jax2onnx-full-v2-bounded-repairs-v1/ROOT-configured-typing-v4-result.json`, `ROOT-style-v4-results.json` |
| Failed joined101 test and observer epochs | `/tmp/jax2onnx-dropout-state-preparation-runtime-v3/joined101-primary-v1-terminal-failed.json`, `/tmp/jax2onnx-dropout-state-preparation-runtime-v4/joined101-primary-v2-terminal-failed.json` |
| Reviewed public-call test correction | `/tmp/jax2onnx-public-legacy-test-correction-astra-v1/verification.json` |
| Joined101 v5 prerequisites only | `/tmp/jax2onnx-joined101-v5-astra-review-v1/verification.json` |
| Actual primary-full/secondary integration and applied inventory | `/tmp/jax2onnx-mlp-normalization-MAIN-actual-integration-astra-v3/verification.json` |
| Seven actual configured hooks | `/tmp/jax2onnx-mlp-normalization-MAIN-integration-v3/ROOT-actual-configured-hooks-v1-terminal.json` |
| Actual joined101 dual execution | `/tmp/jax2onnx-dropout-state-preparation-runtime-v5/joined101-dual-v3-closed.json` |
| Independent finite dual result ACCEPT | `/tmp/jax2onnx-joined101-actual-result-astra-v1/verification.json` |
| Verified 42-path code commit and seven commit hooks | `/tmp/jax2onnx-mlp-normalization-MAIN-integration-v3/ROOT-code-commit-v1-verified.json` |

These local receipt paths support review; tracked code has no dependency on
them. The full B–F contract remains open: all existing functionality, remaining
models and optional environments, wider scope/export combinations, and evidence
for claimed browser or other execution targets must be addressed before the
public transition. The final ordinary `to_onnx(...)` call has one source-preserving
contract, without a public profile choice or primitive-specific legacy fallback.

## What is implemented

[`PrecisionPolicy`](../../../jax2onnx/converter/precision_policy.py) separates
LEGACY and SOURCE. The [shared private harness](../../../tests/extra_tests/converter/precision_test_support.py)
uses the actual tracing helper, IRContext, dispatch, plugins, optimizer,
post-processing and serialization. [source_export.py](source_export.py) is a
compatibility entry point for the internal demos. The bounded harness lives in
test support; it is not a second public export API.

SOURCE preserves traced output dtypes and constant payloads in the audited
paths and suppresses late legacy promotion. Child contexts inherit the policy
and recursive validator. The caller's ambient x64 setting is retained; typed
FP64 inputs under disabled x64 are rejected before tracing.

Both input normalization steps preserve `weak_type`. For scalar `2048` plus
explicitly FP16 `1`, weak inputs return FP16 `2048`; strong inputs with the same
shape and dtype return FP32 or FP64 `2049`. Tests use the original JAX values,
arrays and typed specifications, both x64 states, serialized CPU output, and
actual legacy calls. Shape-only source inputs use the proposed strong FP32
default. Legacy normalization keeps its observed compatibility behavior.

Selected Function and If paths preserve mixed FP64/FP32 outputs, direct and
shared rounded values, static Boolean/string arguments, captures, repeated
results, and tuple/list/dict returns. Output aliases and formal-input
passthrough receive distinct Identity outputs where required by function
signatures. Weak/strong inputs specialize separately. Both If branches are
selected at runtime in the serialized-artifact tests.

## Review findings

The initial 168-case internal suite passed on both stacks but missed the
following cases. The repairs now have durable tests in the regular suite.

### Local function settings must survive body retracing

With ambient x64 disabled, a model called this decorated function inside
`with jax.enable_x64(True)`:

```python
@onnx_function
def cancellation(x):
    wide = x.astype(jnp.float64)
    return ((wide + np.float64(1)) - wide).astype(jnp.float32)
```

For FP32 input `1e8`, JAX returned FP32 `1`; the exported function returned FP32
`0` on both installed runtimes, with optimization off and on. Reversing the
local and ambient x64 settings reversed the discrepancy. A wide-output variant
instead failed the checker. Matching output types could not catch the silent
internal change.

The source call now records x64 and default matrix precision in typed metadata,
replays those settings for body tracing, and includes them in specialization.
Existing typed formals are reconstructed without truncating an already-created
FP64 value; the callee body still runs under its captured settings. Result avals
are checked again after retracing. Tests cover same-signature calls with local
settings, nested functions, failure restoration, and matrix settings whose
explicit algorithm must be rejected rather than lost.

The [context regressions](../../../tests/extra_tests/converter/test_source_function_context.py)
cover x64 and default matrix precision. They do not establish preservation of
every possible JAX setting or concurrent export.

### Cached branch traces must not cross export policies

A conditional branch containing an `onnx_function` could keep the SOURCE
primitive in JAX's trace cache. A later legacy export of the same callable
failed with an unexpected internal output-metadata keyword. Direct
source/legacy/source tests did not exercise that cached branch path.

A private [compatibility helper](../../../jax2onnx/_compat/jax.py) now makes
ordinary execution, LEGACY export and SOURCE export distinct JAX trace/JIT
cache contexts. It restores the prior context after success and failure;
no global cache clearing is used. Both export orders, omitted/false/true
legacy calls, cached conditionals, ordinary eager/JIT execution, and failed
export followed by ordinary execution are checked. The JAX State constructor
used at this compatibility boundary is available in the declared minimum
[JAX 0.8.1 source](https://github.com/jax-ml/jax/blob/jax-v0.8.1/jax/_src/config.py).

### Algebraic fusion must respect the source operation sequence

Admitting Div exposed a legacy rewrite: `(x+x)/float32(2.00001)` became Mean and
returned `1` instead of approximately `0.999995` for `x=1`. SOURCE now retains
Add and Div, as well as separate Mul, ReduceSum and Sqrt operations where
traced. Unvalidated Mean, LpNormalization, ReduceSumSquare and ReduceL2
substitutions remain disabled under SOURCE. LEGACY retains its existing rules.
[Arithmetic regressions](../../../tests/extra_tests/converter/test_source_arithmetic_fusion.py)
check the divisor, overflow, shared rounded values, zero-row behavior and
source/legacy isolation. Removing a converter fusion does not repair a runtime
rounding failure by itself.

### Durable tests now run in the normal CI collection

Input and scope regressions moved from this documentation folder to `tests/`.
New context, matrix, normalization, arithmetic and sample-acceptance regressions
also live there. The existing CI's default pytest collection includes them;
no workflow change was needed. Internal source/UX modules are compatibility
entry points to shared test helpers, and the demo commands still work.
Supplemental PoC diagnostics remain explicitly invoked outside `testpaths`.

## Follow-up review on 2026-10-18

The Astra Ultra review on 2026-10-18 found two additional SOURCE defects after
the preceding checkpoint. Both were implemented with 6.1 Sol Ultra and reviewed
again independently with Astra Ultra, which found no remaining blocker in this
bounded milestone.

With `JAX2ONNX_ENABLE_REWRITE_MUL_RSQRT=1`, FP32
`x * (float32(1.000001) / sqrt(y))` at `x=y=1` became `1` instead of
`1.0000009536743164`. The finalizer now passes its explicit policy to the shared
optimizer. Each registered pass must declare eligible policies; one selection
covers the existing top-level and Function paths. SOURCE excludes reciprocal
reassociation, Swish fusion, ConvTranspose bias folding and transpose/reduction
movement pending numerical audit. Structural cleanup remains enabled. LEGACY
retains its defaults and environment-controlled behavior. See the
[32 optimizer regressions](../../../tests/extra_tests/converter/test_source_optimizer_precision.py).

The function-cache defect involved two registered pytree instances storing
FP64 bias `1e8` with identical shape and bytes but different `weak_type`.
For `(x + bias).astype(float64) * float64(2**-20)` with FP32 input `1`, JAX
returned `(95.367431640625, 95.36743259429932)`; one reused ONNX Function
returned the first value twice. SOURCE keys now retain promotion metadata
recursively through supported static containers, captured arguments and pytree
instance state. The extension applies at primitive argument packing, static
capture keys and unique instance keys. Immutable JAX snapshots retain their
metadata, legacy keys are unchanged, and equivalent states still share a
definition. The [11 captured-state regressions](../../../tests/extra_tests/converter/test_source_captured_precision.py)
cover both orders, nested static arguments, original JAX references, saved
artifacts and both runtime profiles. Removing the new key reproduces four
failures, including the nested-static-argument cases.

## Numerical implementation and acceptance

| Case | Current local result | Limit of the evidence |
| --- | --- | --- |
| FP16 matrix operands, preferred FP32 result | SOURCE emits local FP32 operand casts and FP32 MatMul, returning `2049` for the witness while preserving original FP16 bindings. | Positive static rank-2/rank-3 standard matrix axes and default precision only. Other precision algorithms/types/shapes are explicitly rejected. No performance claim. |
| FP32 squares, selected FP64 variance sum | The actual JAX callable exports inline, in a Function and in both If branches. Exact variance is `8388609`; constant-row output is zero. | Selected fixed CPU samples; independent control remains separate. No general normalization or legacy accuracy-policy adoption. |
| Original FP16 square then sum | The source graph retains Mul then ReduceSum. ORT 1.29 and 1.31 still return zero squares but sum `2**-23`. | Failed numerical acceptance under both tested optimization settings. No automatic repair is established. |
| Explicit wider arithmetic with FP16 intermediate cast | The changed-source sample returns the expected zero squares and sum. | Sample agreement only; not an automatic transformation or equivalence proof. |
| FP32 `x * (1 / sqrt(y))`, `x=3`, `y=2` | Saved source graph and reference retain `2.1213202476501465`. ORT 1.29/1.31 match with optimization off; optimization on returns `2.1213204860687256`. | Failed acceptance under the selected exact-sample criteria with optimization on. Converter preservation does not prevent runtime reassociation. |

An additional ad-hoc probe uses FP16 input `1`, weak FP32 bias `4096`, then
converts the sum to FP32 and scales by `2**-20`. JAX eager/JIT and
ReferenceEvaluator return `0.00390625`; ORT 1.29/1.31 return
`0.003907203674316406` with optimization off and on, losing the intermediate
FP16 Add rounding. This is distinct from square/sum and from successful cache
specialization. The positive x64-off cache regression instead chooses an
epsilon from JAX's promoted dtype and performs FP32 arithmetic. The ad-hoc
failure remains numerical evidence, not a durable acceptance regression.

The exact-one optimizer tests allow a future runtime correction to pass.
Returning the reciprocal as another output preserves the chosen sample under
both current ORT profiles; this is an observation, not a proposed output-interface
change. No automatic runtime repair is established.

The optimizer audit also found an existing LEGACY limitation: moving a
Transpose around ReduceMean with explicit axes inside a Function registers an
initializer that is not serialized as a Function node. Checking then fails
with a missing axes input. SOURCE excludes the pass and its explicit-axis
Function regression passes. The legacy defect remains a separate inventory
item; an invalid artifact is not an accepted compatibility result.

The [CPU sample gate](../../../tests/extra_tests/converter/precision_runtime_support.py)
records the actual ORT version, CPU provider, architecture, optimization and
thread settings, and checks exact output types, shapes and values.
`require_sample_acceptance` rejects FAILED and UNVERIFIED results. Kernel
load/run errors are separate. A future runtime correction may satisfy the
original FP16 sample; a diagnostic that recognizes the current discrepancy
cannot pass acceptance. A report covers only its supplied artifact and samples,
not all model inputs or a deployment generally.

The [numerical findings](numerical_contract_findings.md) describe the unchanged
public matrix gap, new source implementation, controls, runtime failure and
rejection evidence in detail.

## Audited exploration boundary

The harness admits static floating inputs at opsets 23 and 26. Primitive paths
include add/sub/mul/neg, numeric conversions, ReduceSum, equal-width BitCast,
floating gt and the Boolean-to-INT32 conditional selector, two-branch cond,
real onnx_function calls, Div, BroadcastInDim, lax/NumPy Sqrt and the bounded
matrix lowering above. BitCast requires explicit opset 26; different widths
are rejected. Function and branch bodies are checked recursively.

An allowlist is not a support matrix for all compositions. BF16 structural
checks do not establish CPU kernels. JIT regions, loops/scan, symbolic shapes,
effects and unaudited operations remain excluded. Public names/layouts,
parameter materialization and return-mode integration remain outside this
harness. There is no public deployment-target API, generic capability engine,
automatic fallback selection, latency benchmark or memory benchmark.

## Recorded validation

Installed CPU stacks used for these checks:

| Python | JAX / JAXLIB | NumPy | ONNX | ONNX IR | ONNX Runtime |
| --- | --- | --- | --- | --- | --- |
| 3.12.9 | 0.11.2 | 2.5.4 | 1.23.2 | 1.0.0 | 1.31.0 |
| 3.11.12 | 0.10.2 | 2.4.6 | 1.22.0 | 1.0.0 | 1.29.0 |

Durable SOURCE suites pass on both stacks: input properties (46), function/If
scopes (16), call-context/cache isolation (35), preferred-FP32 matrix products
(35), wider normalization (6), arithmetic boundaries (12), and runtime sample
acceptance (9). These 159 cases belong to normal pytest collection. The
supplemental PoC suite contains 107 cases, including expected exclusions and
recognized diagnostics. Passing it is not a numerical certificate.

The follow-up adds 32 optimizer and 11 captured-state regressions to normal
collection, bringing durable SOURCE coverage to 202 cases. Each new suite
passes on both installed stacks.

Final local validation on the updated checkout, 2026-10-18:

- Full repository suite on Python 3.12/JAX 0.11.2: **4662 passed**, using CPU
  and four workers.
- Python 3.11/JAX 0.10.2 focused converter/architecture suite plus supplemental
  PoCs: **616 passed**, using CPU and four workers. This is not a second full
  repository run.
- Supplemental PoCs on Python 3.12: **107 passed**; those same cases also pass
  within the Python 3.11 run above. The numerical demo runs and reports the
  original FP16 sample as rejected by numerical acceptance.
- Repository typing: **538 source files** pass. The four private test-support
  modules also pass a separate mypy check.
- Ruff, IR-builder policy, Black for all **35 changed/new Python files**, and
  the strict MkDocs build pass. Internal Markdown rendering and **143 local
  links/anchors** pass separately because these documents are outside the
  published documentation build. `git diff --check` passes.

The formatting result covers touched files; the preceding implementation
checkpoint recorded pre-existing formatting differences elsewhere in the
repository. These local results establish neither a remote CI result nor a
published capability.

## Remaining migration work

The [next work](../precision_migration_plan.md#next-implementation-work) defines
and enforces a finite initial subset, integrates the existing export boundary
privately, and connects validation evidence to caller-facing diagnostics.
The failed runtime samples remain excluded from acceptance under their recorded
criteria and profiles. Broader operator/scope/target coverage and any public
selection still need their own acceptance. These changes advance Stages 3 and 4 without
completing the migration.


## Exact floating selection

The later two-file SOURCE update repairs the already-admitted Boolean-selector,
two-arm FP16/FP32/FP64 case. It stacks the arms in `[false, true]` order, casts
only the selector to INT32 values 0/1, and copies the chosen value with
`GatherElements(axis=0)` and `Squeeze`. A scalar selector expands to the actual
arm shape. There is no floating cast or arithmetic in this selection. Canonical
arms, siblings and captured values remain intact. Native Where forwarding,
profile admission, public LEGACY behavior, report schema and existing
BOOL/integer/BF16/complex/multiway boundaries retain their meanings. BF16
structural evidence still does not establish CPU kernel support.

Independent review accepts the exact two production/test postimages and the
bounded owner observations. Both supported stacks pass 179 owner cases,
including 38 new cases. Before the repair, the final signed-zero regression
fails on original production on both stacks; afterwards it passes under the
same strict criteria and native reference bytes. The retained witness reports
record twelve converted strict successes per candidate stack, with six direct
old-Where failures and six separate positive-zero diagnostic-reference successes.
Those diagnostic successes do not claim native equivalence. Other focused cases
cover tested NaN payloads, infinities/subnormals, native promotion/broadcasts,
scalar and empty results, B=2/5 with repeated/complementary masks through one
artifact, and Function/If captures and shared outputs. Passing empty-output
cases does not separately prove execution of an empty GatherElements kernel.

The saved production witness evidence contains native arrays, ONNX models and
validator output hashes; it does not retain raw actual target-output arrays for
all focused cases or establish a total runtime-call count. Export reports retain
runtime/numerics UNVERIFIED until separate sample validation. The old Where
artifacts and failures, original callback contracts and strict criteria are
unchanged. The operator-documentation list adds GatherElements; testcase
registration metadata stays identical, while whole-file source provenance
naturally changes.

This representation has a substantial memory cost. The independently reviewed
standalone comparison recorded 576 target calls and 168 sessions: 504 small saved
outputs and 72 realistic A/B/A observations bound to four distinct actual arrays.
At FP32 `[1,12,512,512]`, named logical buffers total 48 MiB for INT32 indices,
versus 60 MiB for INT64 indices and 45 MiB for the tested Compress alternative.
INT32 gave the better measured speed/memory tradeoff in those finite runs. The
2048 attention example still implies 768 MiB of named buffers, including a
384 MiB arm stack, before broadcasts. Process high-water memory includes runtime
arenas, inputs/references and other work; three-call observations are not an
operator-allocation or general-throughput guarantee. No speculative scalar fast
path or memory-cost waiver is part of this change.

The first integrated full run retained all 8,810 expected identities and ended
with 8,721 passes and 89 setup errors, with no test failures or skips. Every
setup error came from the old strict Gather fixture rejecting the new floating
selection representation. That failed run remains immutable; it did not execute
a failing Gather numerical comparison.

The reviewed three-path tooling repair gives the current Gather representation
its own exact 26-node/14-constant template. The historical 20-node/13-constant
template stays literal and is rejected by the current revision. Edge order,
attributes, dtypes and constant bytes remain strict; the original 90 Gather
cases are retained and nine new guards are added. Because the current Dot0
control binds the complete oracle source file, its one current literal hash is
updated to that exact revision. The control identifier and meaning are unchanged;
historical descriptors and failure records are not repinned. There is no
permissive graph fallback or relaxed numerical criterion.

Both MAIN artifact runs preserve the four ordinary controls and repeat all 26
selected cells: 25 diagnostic artifacts plus the unchanged Gather x64-off FP64
configuration conflict. All twelve replay invariants pass, with unchanged
3,578 unselected cells and 126 pending cells. The fresh references and artifacts
match the immediate reviewed owner epoch on both stacks. Gather's original
source, native trace, inputs and references remain bound to the same saved
artifact: the raw callback still fails while SOURCE structure is VALIDATED.
Product lowering, runtime and numerical acceptance remain UNVERIFIED, criteria
and optimization remain None, and these artifact runs attempt no ORT execution.
This does not revise the separately recorded historical matrix-reference issue.

Independent review accepts the exact six-path MAIN integration: two selection
paths, three strict-tooling paths and the manual inventory. Both focused runs
pass 1,911 cases/5,733 phases across eighteen collected modules and fifteen
recorded MAIN imports. The five effective/hidden JAX settings, policy and 25
watched callables are restored; all 871 Python hashes, nineteen checkpoint paths
and 29 Markdown files stay exact. Style, configured typing and both inventory
CLIs pass. The inventory records 220 manual files, 1,372 functions and 620
migration additions, retaining all 1,363 immediate predecessors and 752 original
anchors, zero removals, and 1,802 raw rows/3,604 cells. The fresh complete primary
suite passes exactly all 8,819 expected identities, the earlier 8,810 plus nine
Gather guards, with no failures, errors, skips or omissions. It records 13,773
warnings and 682.91 pytest seconds (682.845 XML seconds); all 871 Python hashes,
nineteen checkpoint paths and 29 Markdown files remain unchanged. Code commit
`778dce3c5f7e799f3288199cd5cdc13b51bb03ed` then passes all seven installed hooks
without bypass, preserving those hashes and committing exactly six changed
paths. The recorded worktree is clean after that commit. The preceding
8,772-test/code-8f checkpoint and the failed first selection/Gather full run
remain historical evidence.

The older-stack FP16 decoder benchmark remains an independently established
baseline failure. This repair does not close it, general selection/provider
support, unchanged MLP training/state behavior or the full precision migration.
At that checkpoint, the separate MLP work had native preparation, a measured
PRNG ownership review and floating criteria frozen before target runs. Its
implementation was outside MAIN and model runtime/numerical acceptance was
pending. The later finite result is recorded in the current acceptance section
above.

| Frozen evidence | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-where-cheaper-probe-astra-review-v1.md` | `2bb8435040ae9fe47d20404402a8f70ba9765e56d8d2b660e368d4684cd94a52` |
| `/tmp/jax2onnx-exact-select-astra-review-v1.md` | `fbcc14994faf36fdc9f8f364c0d0ac092dc513186491df89244b32a06d9e67c6` |
| `/tmp/jax2onnx-exact-select-main-integration-freeze-v1.json` | `5e9b3c140a32ad23c60dbba85cdd6b5f0ec6d03c59b70c0712421a49f36f8a66` |
| `/tmp/jax2onnx-exact-select-main-focused-dual-v1.json` | `555567a2a0f938ecfe3505c80c3e5d02abb7aac2b4390244bc17b01081abf885` |
| `/tmp/jax2onnx-exact-select-main-checks-v1.json` | `fec9aef745cb037f18beab9b6a9f2dad96e6d91d4f1b17a65bb6a291a440bac9` |
| `/tmp/jax2onnx-exact-select-main-full-result-v1.json` | `7dd9ab74cf73fbab7c5538e1e0433307bd1f613b0023ebf674ef638975834847` |
| `/tmp/jax2onnx-gather-selection-code-astra-review-v1.md` | `1163ea5defc57fb39fdd3abce33c6421576becadcaaec45a7350a903096e5b12` |
| `/tmp/jax2onnx-gather-selection-postrun-astra-review-v1.md` | `7bf282c5e6f1cbde0451c3fe063459bc7d614313d50950fd55fbf3ab476c90bf` |
| `/tmp/jax2onnx-gather-selection-live-control-release-v1.json` | `2937b4fafe647fe132f2e5e602e314cd31c3934e43ef7efc79145cdb10f3dbfe` |
| `/tmp/jax2onnx-exact-select-main-integration-freeze-v2.json` | `8411e5cc4c4cd8195f840ae91f9c79fe250a14703c252a257b2a31ff30ac8f1a` |
| `/tmp/jax2onnx-exact-select-main-focused-dual-v2.json` | `c60707b5d4efb8fcffa32382f0eefbe2cbd755cdb826894bfb1a6a1a595517f2` |
| `/tmp/jax2onnx-exact-select-main-checks-v2.json` | `b7f074374f8694963761093474be4048cc6b517014599c8d7fe565fb6dfbe13d` |
| `/tmp/jax2onnx-exact-select-main-artifacts-dual-v2.json` | `aab46edf220965f16bc74109d5bac5b7bee793e456be42eaef09796c4708fe85` |
| `/tmp/jax2onnx-exact-select-main-integration-astra-review-v2.md` | `542ff2a5b9e42a370d6cdcd70ee2f5a51099dd82dfd3ceaa68363045d9decdd5` |
| `/tmp/jax2onnx-exact-select-main-integration-astra-verification-v2.json` | `23ade31437667b95bfca9a02af7f1d1e0f1263aa9073cdc12cae41cb763bf21e` |
| `/tmp/jax2onnx-exact-select-main-full-result-v2.json` | `a05370d277f5e208193085c8a7b8c7b653d0bdc606abf361c0a970cdfa3b3faa` |
| `/tmp/jax2onnx-exact-select-main-code-checkpoint-v2.json` | `4942d59f8a6da92a97db2af0b67ed734ac766170d5a58d244bf961aa09aec441` |
| `/tmp/jax2onnx-exact-select-main-code-commit-v2.log` | `2ab88d93bfbae13995435ea6118fb1ec702a19f4c654aa6bd703611980a26db6` |


## LayerNorm and mean native forwarding

This section records the preceding forwarding-only checkpoint. The later
[MLP and bounded normalization result](#mlp-state-and-bounded-normalization-acceptance)
has separate acceptance and explicit limits.

Independent source/evidence review accepts the finite five-path forwarding
checkpoint. The three framework LayerNorm wrappers and `jax.numpy.mean` now call
the captured original exactly once under private SOURCE policy, before legacy
argument/default processing, input coercion, configuration reads or axis/mask
handling. Original argument identity, omission, exceptions and signatures remain
observable. Complete LEGACY bodies, all four registration ASTs, six raw mean
cases/callbacks, lowerers, abstract evaluation, batching and JVP code are unchanged.
There is no new profile admission, public API change or public precision mode.

The retained NNX witness has FP32 inputs and FP16 affine parameters. Native JAX
first promotes the parameters to FP32. Before the repair, the activated wrapper
instead converted the FP32 input to FP16 and emitted synthetic `nnx.layer_norm`.
The final full-plugin-world trace matches the original native graph on both
supported stacks. This is a concrete removal of exporter-imposed narrowing.

Full-world Equinox testing exposed one additional prerequisite: its native
`reduce_sum → broadcast_in_dim → div` mean was replaced by synthetic
`jax.numpy.mean`. Forwarding mean through its own captured original repairs that
interception. The initial suggestion of an independent var/JIT divergence was
incorrect: complete nested `_var` JIT parameters already matched; only outer SSA
numbering changed after mean collapsed three equations to one. The retained
correction addendum supersedes that diagnosis. No var/JIT or profile change is
part of this checkpoint.

The owner runs pass **127 tests on each supported stack**, retaining all eighteen
previous full-world Equinox failures: sixteen dtype/affine combinations, the
weak-scalar/context case and symbolic batch. Sixteen isolated wrapper-only cases
remain supplemental. The strict-Flax correction compares the actual native
NNX/Linen `TypePromotionError` and message rather than requiring a native success
that does not occur. Failed epochs and the five direct-mean pre-repair failures
per stack remain recorded.

Fresh native-versus-activated comparisons cover typed traces and nested bodies,
actual constants, eager outputs and fresh-JIT outputs, including exact payload
bytes. Inputs, parameters and computation dtypes, fast/slow variance, epsilon,
axes, masks/where, affine options, weak values, contexts, state/key arguments and
native errors are exercised. State snapshots, five global/effective settings,
source bytes, LayerNorm/mean callable identities and policy restore. ORT
constructors and execution entrypoints are denied throughout these owner checks.
The source-derived phase totals are 132 trace, 129 eager and 129 fresh-JIT pairs,
including 32 supplemental pairs per phase; these are test contracts, not separate
observed phase receipts. Separate concrete full-world captures retain three equal
native/activated traces per stack. Existing LEGACY/IR checks pass 30 cases per
stack plus all six original mean graph-checker/callback anchors.

The first MAIN focused selection on JAX 0.11.2 retains **696 passes and one
failure among 697 cases**. The failure is the fixed-source preparation test's
stale expectation of eight current barriers. Native mean forwarding now exposes
already admitted reduction/broadcast/division/subtraction operations in all four
minimum-precision examples. Their FP16/BF16/FP32 inputs produce FP32 outputs;
FP64 produces FP64, all at shape `[1,4]`. Actual artifacts and representation/
lowering reports are saved, while runtime lists stay empty and target numerical
acceptance stays None. Their bitwise native reference criterion is unchanged.
Only the two protected-sum and two HIGHEST-dot barriers remain in that fresh run.
The original twelve-export/four-session/four-run schedule still executes only the
two exact target programs. The independently accepted companion-test repair
updates only the stale live barrier assertions and adds checks for all four
admitted artifacts: native traces and references, report contents, tensor-port
dtypes/shapes, ONNX schema, reload and serialized bytes. Both owner stacks pass
all eleven existing module cases with unchanged identities. No additional target
program runs were added. The original failure, the PoC's literal historical eight
barriers, source programs, inputs, bitwise native criteria and older findings are
retained. This additional mean export progress does not establish LayerNorm
export or new runtime/numerical support.

At this forwarding-only checkpoint, default-audited SOURCE LayerNorm export
stopped at these native operations on both stacks:

| Framework | First barrier at that checkpoint |
| --- | --- |
| NNX | `square`, equation 2 |
| Linen | `square`, equation 0 |
| Equinox | `square`, equation 3 / JIT body / equation 4 |

The profile was unchanged at that checkpoint. These barriers identified
implementation dependencies, not ONNX impossibilities. Original Functions014/015
and whole-model export/runtime/numerical acceptance were **UNVERIFIED**. The
separate MLP implementation was then outside MAIN; its later finite acceptance
and the subsequent bounded normalization result are recorded above. Complete
Functions014/015 acceptance remains open.
The final goal remains every existing capability through one ordinary call that
preserves the source computation, including explicit external JAX preparation
for fixed-source workflows; no permanent public mixture of precision policies
is introduced.

The five reviewed forwarding/test paths, the existing companion test and the
manual inventory update form the seven-file MAIN integration committed at
`ae1d61d1`. Independent review confirms exact accepted postimages across the
872 Python files and preservation of all 29 precision Markdown files during code
validation and commit. The inventory contains **221 manual files, 1,385 functions
and 633 migration additions**. All 1,372 immediate predecessor functions and all
752 original anchors remain, with no removals. Only the new test module's thirteen
functions were added. The 1,802 raw rows, 3,604 cells and all nonmanual fields are
unchanged.

| Frozen evidence | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-layer-norm-mean-native-forwarding-candidate-v1/SHA256SUMS.json` | `7f4c06ae23d3763df62b8fe70d712fc53685d35970e9c92a898c9bfda79dd6ca` |
| `/tmp/jax2onnx-layer-norm-mean-independent-astra-review-v1.md` | `d39f6dd27ab3ee7315ff0aaffd5ebfea6b2e6befd27020e7a637f985b38f293b` |
| `/tmp/jax2onnx-fixed-preparation-companion-candidate-v1/SHA256SUMS.json` | `ab96037ec0c863f35ed1f2eee42c2952f4defcc09c5a3bb2053dcfa5117dfcd0` |
| `/tmp/jax2onnx-fixed-preparation-companion-astra-review-v1.md` | `68d4f8b9df06d60e78d5e4077c1c0df0a18e613fee86cb0281675cd2a9e61913` |
| `/tmp/jax2onnx-layer-norm-main-fixed-preparation-failure-v1.json` | `730b34db1a657d4a4026557b2aa3897dd4696a712ca48022f421c89edc88fcb5` |
| `/tmp/jax2onnx-layer-norm-main-fixed-preparation-failure-v1-closed/manifest.json` | `4ca7cd73584d583ad3146d270bc75a50e4db0a6f00bc7ef9f2045059f53b177e` |
| `/tmp/jax2onnx-layer-norm-main-integration-freeze-v2.json` | `3a68ffcac9d352bcf216a120de3d7687ca7977c2bda67a0a64e0ecd90543474b` |
| `/tmp/jax2onnx-layer-norm-main-focused-dual-v2.json` | `7bcc8b6556e4b91e1b451e8c9e58b74ff0349f5832fe0d8e290c3be0d04707fa` |
| `/tmp/jax2onnx-layer-norm-main-checks-v2.json` | `8ac37ff456beeb15d9899dd68c9ccb2bc401ecfddd55c3ad3cf729cd335f9ab6` |
| `/tmp/jax2onnx-layer-norm-main-integration-astra-review-v2.md` | `6a8715c1e1aecf510be20e3c22bbe119521547beb8d888781b848745158ce2c2` |
| `/tmp/jax2onnx-layer-norm-main-full-result-v2.json` | `1dbbd2fb4ae6de6e80117cb6573de8bd69dc7801f4ae045bbf0f98a814a46678` |
| `/tmp/jax2onnx-layer-norm-main-full-astra-review-v2.md` | `0036b72cccba01480f30d5c92bbb6c5564e14b030af9d5ede920f10690d2e6e8` |
| `/tmp/jax2onnx-layer-norm-main-code-checkpoint-v2.json` | `68b6f539fa687bb535d172290cf61a4a5ecd2e84ce02e64de30c26ff6aed8078` |
| `/tmp/jax2onnx-layer-norm-main-code-commit-v2.log` | `b6d49d1faf06bef9b2d9b42de02a24fef9dfe74701d7af597bc578268c9edaed` |

Validation status: actual MAIN receipts supersede the owner's earlier declared
Ruff/Black/whitespace successes. After the companion repair, both supported JAX
stacks pass the same **697 focused tests**, with **2,091 actually observed
setup/call/teardown reports per stack**. These pytest phases are separate from the
source-derived native comparison totals above. All 31 watched callable identities,
five hidden/global and five effective settings, and the precision policy restore;
21 imports bind each run to MAIN. The refreshed seven quality commands pass:
global Ruff, configured Black version/check for all six Python files, whitespace,
configured typing for 542 files plus registered RNG factories, and both inventory
CLIs. The primary complete suite passes **8,946 unique tests**, retaining all 8,819
previous identities and all 127 new LayerNorm identities, with no missing,
unexpected, failed, errored or skipped cases. Its raw log records 13,609 warnings.
Both the finite integration and actual full-suite result have independent
acceptance. The original 696-pass/one-failure epoch remains retained. Code commit
`ae1d61d14b0122c9f1814c813eba3f495c4365e2` follows predecessor `4f088d82`,
contains exactly the seven accepted paths and completed all seven actual hooks
without skips or bypass. Its terminal receipt records a clean MAIN worktree and
unchanged 872 Python, seven checkpoint and 29 Markdown hashes.


## Native Sum forwarding and measured composition

Independent source/evidence review accepts the native Sum implementation, now
committed with its companion test and manual inventory at `2d1ea944`.
Under private SOURCE policy, the outer wrapper calls its own captured original
exactly once with the original positional and keyword arguments. This occurs
before legacy dtype/axis/default handling, array coercion, `out`/`initial`/`where`
checks or `promote_integers` coercion. Omitted arguments remain omitted; original
return/error identity and signature behavior are preserved. The complete LEGACY
body, all six raw registration cases and callbacks, lowerer/fusion/empty-axis
logic, JVP, batching and custom transpose remain unchanged. This checkpoint does
not change converter core, the SOURCE profile, public defaults or precision flags.

The standalone owner suite passes **90 cases on each supported JAX stack**,
retaining all 70 preparation identities and all eight pre-fix failures. These
are suite cases, including API errors, a captured-callable boundary spy and a
signature check; they are not 90 numerical trace/eager/JIT triples. The coverage
map is source-derived. Seven separately recorded native/SOURCE witnesses per
stack preserve saved typed traces, trace text, closed constants and exact eager
and fresh-JIT payloads. Named-JIT checks reuse the decorated inner function with
fresh outer wrappers, so they do not independently prove fresh inner retracing.
The 22 LEGACY structural/schema/reload artifacts per stack cover six raw anchors,
four fusion/opset cases and twelve empty-axis cases. Their original runtime test
phases remain held. All ORT constructors/runs were denied in these owner checks.

The retained external receipt failures in epochs 1 and 2 came from comparing
current tuple-bearing typed avals with lists restored from JSON. Epoch 3 isolates
ten container differences with identical serialized values. The authorized
epoch-4 correction JSON-normalizes only the current typed manifest for that saved
record comparison; current native
versus SOURCE equality and numerical bytes remain exact. No source, native
criterion or original failure record was changed. Independent review retains the
owner style distinction: Black/Ruff/mypy success stdout exists, while configured-tool
identity and whitespace success lack separate structured process receipts. The
later actual MAIN quality receipts below close that process-evidence gap for the
integrated checkpoint.

The first separate composition used committed LayerNorm/mean code `ae1d61d1`, documents
`21ad27b4` and only the two Sum postimages; it contains no pending MLP source.
Its first **primary-only** run collected 101 tests: **99 passed, one failed and
one was unexecuted**. All 90 Sum tests passed, and 80 actual calls to the unchanged
comparison helper completed. There are 300 observed pytest setup/call/teardown
reports; the native phase semantics remain in the hash-bound helper. The failure
is the existing companion's blanket `profile-validation` assertion at line 192.
The v1 driver correctly stopped because its narrowly defined obsolete-assertion
classifier covered only lines 186/194. Its false acceptance flag is retained;
no source, state or target-schedule guard failed. Secondary was not executed and
no retry or assertion edit was used to relabel that result.

The same four rejected keys now report these actual barriers:

| Cases | Phase/category | Primitive and location |
| --- | --- | --- |
| `exact_original_protected`, `exact_prepared_protected` | `representation` / `representation-limit` | `bitcast_convert_type`, equation 9 in each |
| `linear_original`, `linear_prepared` | `profile-validation` / `profile-exclusion` | HIGHEST `dot_general`, equations 0 and 1 respectively |

Both protected examples request float32-to-uint32 bit reinterpretation. The
unchanged PoC requests **opset 23**; the existing audited representation requires
exactly opset 26 for this BitCast path. An isolated ONNX 1.23.2 schema query confirms
that `BitCast` first appears at 26 and is absent at 23/25. The diagnostic recommends
`opset=26` only if the target supports it and states that export never upgrades
the opset automatically. This identifies the current representation requirement,
not runtime support at opset 26 or an impossibility for every other implementation.
The protected examples remain rejected; this checkpoint does not change their
opset or source to obtain acceptance.

The original twelve native cases and **twelve-export/four-session/four-run**
schedule completed. Eight artifacts were admitted: the two exact programs, two
intermediate-output programs and four minimum-precision examples. All use opset
23. Only the same two exact programs executed, each with disabled/all optimization;
their four target outputs satisfy the unchanged eager/JIT dtype, shape and byte
criterion. The other ten rows retain empty runtime lists and no target numerical
acceptance. All 82 NPY files match the previous accepted companion epoch byte for
byte, including inputs, actual weights, native references and target outputs;
the contract differs only in its epoch identifier. The original native criteria,
including the linear pointwise bounds, and literal historical eight-barrier
`HISTORY` remain unchanged. Sum introduces no new target numerical acceptance.

All 873 Python files, 29 precision and 76 total Markdown files remained unchanged
during that composition run. Five global/local/effective setting values, policy,
61 callable identities including 47 public PRNG callables, and observer wrappers
restore. This is callable preservation, not a separate PRNG-state audit.

The independently accepted existing companion amendment has postimage
`5ccb2e6d75bb0758c8a673bb87b0519f23f78d7b7a2267ecfe8451d9c8f735bb`.
It keeps all four rejected keys and asserts each exact phase/category/primitive,
the protected uint32/opset-26 requirement and conditional/no-auto-upgrade action,
and the linear HIGHEST/default-precision requirement. Original target-output
checks, all four mean artifact/native/report/port/schema/reload/trace checks, the
intermediate/manifest suffix, other tests and original execution limits are exact.
The old driver and failure remain immutable. A new source freeze, strict v2 driver
and separate execution release bind the corrected fixture. Both supported stacks
then pass **all 101 unique cases and 303 observed pytest phases**, with no skip or
obsolete-failure allowance. Each observes 80 completed original comparison-helper
calls, the original 12 native PoC cases and exactly 12 exports/four sessions/four
runs. Both sets of 82 NPY payloads remain byte-identical to the failed v1 epoch;
criteria, protected barriers, eight admitted artifacts and the two original
numerically validated target programs retain the limits above. Five setting
values, policy, 61 callable identities and observer wrappers restore on each stack;
the 47 public PRNG checks remain identity checks only. The 873 Python, 29 precision
and 76 total Markdown hashes remain unchanged during these composition runs.

The four-path Sum code checkpoint `2d1ea944` has parent `21ad27b4`. The MLP
source was outside MAIN and was not a prerequisite for that integration. Default
SOURCE LayerNorm exports then encountered `square`; the later finite MLP and
normalization acceptance is recorded above. Phases B–F and their full
existing-functionality gates remain open. The final goal is one ordinary
source-preserving `to_onnx(...)` call across all existing functionality, including
fixed JAX model sources configured or prepared externally. No public precision
mode or permanent partial-profile choice is introduced.

| Frozen evidence | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-native-sum-forwarding-candidate-v1/SHA256SUMS.json` | `a61a97b9bd7ef3f9fe2ef598e9fa2962da95c831176bf717b52efc39beb93d9e` |
| `/tmp/jax2onnx-native-sum-independent-astra-review-v1.md` | `8c394af1488b458ac25bdf316b686ab008abb4002e8214fe966e31ad0c99eb81` |
| `/tmp/jax2onnx-native-sum-composition-primary-failure-candidate-v1/SHA256SUMS.json` | `d2a13a82178581c974faf7b3c68469df26ea876726f445ded23f90c3975b1d97` |
| `/tmp/jax2onnx-native-sum-composition-primary-failure-candidate-v1/summary.json` | `7daa4fccacff901fb5a8acacbcef59761a0811f1fbfb34712648e5c30ce454f8` |
| `/tmp/jax2onnx-native-sum-composition-astra-review-v1.md` | `2811284b28bea3a6d23d07ad64d9e3d01fd192cc5b6b5b8c2969b1b03f2d7937` |
| `/tmp/jax2onnx-native-sum-composition-onnx-schema-astra-v1.json` | `774ae7d4aed1631e1249c7015f773e37ad6923488782735b6c0271e260257b1b` |
| `/tmp/jax2onnx-native-sum-companion-candidate-v1/SHA256SUMS.json` | `86af7425afa2f0f0ac598c3910a6510bd6cf9931cbf7d0f318df5efcd1739a58` |
| `/tmp/jax2onnx-native-sum-companion-astra-review-v1.md` | `aa84e9a069f3750fffd0610b3a644a07bc6ddcf018901b5e65c54a260f74e7bc` |
| `/tmp/jax2onnx-native-sum-composition-measurement-v2/SHA256SUMS.json` | `b8b8a4f4a8f7efe160aa3c7d175600ce702cf14ab1710133ee3ea0fdd6c0ed72` |
| `/tmp/jax2onnx-native-sum-main-integration-freeze-v1.json` | `4af1cc1045a60136aa04a2e47e025e03df047099fe664b78befeb66214360cad` |
| `/tmp/jax2onnx-native-sum-main-focused-dual-v1.json` | `14f6c67c86a4ecfd0f2134b00ddf4105986cc6728647cca8a6a1dce92669ddea` |
| `/tmp/jax2onnx-native-sum-main-checks-v1.json` | `ad077a5dd71811f398c5d4c08eb0174b3b01355ca289526128763bfeae7748fc` |
| `/tmp/jax2onnx-native-sum-main-full-result-v1.json` | `a9b58ee47dd25bb0915c7656a61f0e157f4a1cb1c47f70f4de992a01b1c0c2a1` |
| `/tmp/jax2onnx-native-sum-main-integration-astra-review-v1.md` | `9eada04c14b7005fd2e5a8cef618d1308ab7faa4798942e04ca02a03e4942a72` |
| `/tmp/jax2onnx-native-sum-main-code-checkpoint-v1.json` | `5d5933ed0b85d814a9940dbfd2590a09f2e7c67d9320067387eb3a5fe3f4e6f8` |
| `/tmp/jax2onnx-native-sum-main-code-commit-v1.log` | `28bd9e2de62d60bb406e310a35e9f190506157c02ca989747332213c6e6118c1` |
| `/tmp/jax2onnx-native-sum-main-apply-enumeration-failure-v1.json` | `7faf06d3a833a11fc9cd545b0a04257af07b6ed2e327c63c852cf3c653f37b9e` |

Validation status — completed code checkpoint: independent review accepts the
four-path MAIN integration (`sum.py`, its new test, the existing PoC companion
and handwritten inventory). The inventory has **222 manual files, 1,401 functions
and 649 migration additions**, retaining all 1,385 predecessor functions and all
752 original anchors. Exactly 16 new test functions are added; all 1,802 raw
rows/3,604 cells and nonmanual fields remain unchanged. Both MAIN focused runs
pass **787 cases and 2,361 observed setup/call/teardown phases**, with all 33
watched callables, five hidden/effective settings and policy restored; 23 recorded
imports resolve to MAIN. Seven actual quality commands exit zero: global Ruff,
Black version/check, whitespace, configured typing plus RNG registration, and both
inventory CLIs. Typing checks 542 source files. Configured Black revision 25.1.0
reports version `0.1.dev1+g8a737e727`; the secondary inventory preserves the existing
absent-on-this-stack `one_minus_square` baseline entry. The fresh complete primary
suite passes **all 9,036 unique expected identities**: the previous 8,946 plus
exactly 90 new Sum cases, with no failures, errors or skips (13,609 warnings;
545.74 pytest seconds). ROOT's earlier apply-enumeration assertion, caused by
ignored debugvenv files, remains recorded; corrected Git enumeration verified
the identical already-applied sources without changing a criterion. Code commit
`2d1ea944adb17f477b76f4951d74aa59cc028cdb` contains exactly the four accepted paths
and passes all seven actual repository hooks without bypass. Its terminal receipt
records a clean MAIN worktree and exact 873 Python, four checkpoint and 29 Markdown
hashes. At that Sum-only snapshot, its companion documentation was a separate
external draft awaiting review, application, build/checks and its own commit.
Those completed code gates do not establish new target/model
numerical support or complete phases B–F.
