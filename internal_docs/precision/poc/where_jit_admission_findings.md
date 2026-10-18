# Native Where and Typed JIT Admission Findings

Status: bounded private semantics and actual-main integration independently
accepted and committed at `ae820b1a`, 2026-10-18. The eleven-path implementation preserves native SOURCE
`jnp.where` calls and adds typed selection dependencies and pure closed JIT
bodies to the existing audited profile. Before the shared x64 restoration repair,
both actual-main selections pass 2,661 cases; the independent selection passes
200 cases per stack. The separate two-path restoration repair is independently
accepted. Both post-repair main selections
pass 89 cases with hidden x64 True; five actual MAIN imports, all 863 Python
hashes and restored state are verified. Independent integration review accepts
these results. The fresh complete primary-stack suite passes
**7,236 tests** (13,130 warnings, 400.95 pytest seconds), with no failures, errors
or skips. All fourteen checkpoint paths and all 863 Python hashes stay exact.
Code commit `ae820b1a` passes all seven actual repository hooks; the tested
checkpoint hashes remain unchanged after commit.
Public LEGACY precision policy, the `source-static-v1` identifier and report schema 1
remain unchanged. This checkpoint does not complete the precision migration.

The preceding [Exp/Mul checkpoint](pointwise_mul_exp_findings.md), code
`7c68300b` and documentation `58ff66e0`, retains its 6,933-test full-suite result.
The [Leading-B checkpoint](leading_batch_matrix_bias_findings.md) retains its
6,816-test result and separately accepted forty FP32 runtime calls per stack.
Its failed preparation attempts and post-return wrapper exit-1 reconciliation
remain recorded; no numerical retry or historical relabeling occurs here.

## Native calls and bounded representations

SOURCE forwards the original Where arguments exactly once to captured native
JAX before the LEGACY parser. Native errors, promotion, weak typing and explicit
broadcasting therefore belong to the same original call. The custom primitive,
abstract evaluation, lowerer, differentiation rules, registration and helpers
retain their prior AST; LEGACY dispatch keeps its existing behavior. Native
one-argument or fixed-size tracing is not newly admitted product support.

| Representation | Audited boundary |
| --- | --- |
| Static Boolean dependencies | Positional BOOL inputs, explicit BOOL broadcast and same-BOOL conversion, with exact typed shapes and weak properties. Symbolic BOOL inputs remain excluded. |
| Two-case selection | Preserve native selector/false-arm/true-arm order. Floating arms/results retain their same dtype; BOOL results use logical selection. Result weakness follows both arms, independently of selector weakness. |
| Floating numeric condition | Only the native static floating comparison against a same-dtype scalar zero literal; arbitrary numeric comparisons are not admitted by this rule. |
| Leading-B floating selection | Static scalar BOOL selector and equal floating arms/results using the same actual native symbolic scope, within the existing finite B contract. |
| Pure JIT carrier | Exact eleven-field native parameter schema, actual closed body, recursively validated operations, exact typed ports and closed constants, and supported native inline/placement forms. |

BOOL selection uses `Not`/`And`/`And`/`Or`, including correct intermediate shapes;
floating selection uses a same-type ONNX `Where`. Outputs are fresh values,
so input aliases and captured constants cannot become writable outputs. No
implicit arithmetic Cast or dummy operand replaces the source computation.
Native empty BOOL cases can contain no select operation at all. The four old
structural assertions now require that observed native alias case while keeping
their inputs, outputs and runtime checks.

The JIT validator checks actual input/output counts, dtypes, shapes, weak types
and symbolic scope identity. Closed constants must have the declared count,
type and shape; inspecting their values does not recanonicalize FP64 under an
ambient x64-off setting. Effects, high/ref forms, cyclic nesting, duplicate
bindings, unbound uses and unbound outputs are rejected before emission.
Literal/constant/formal aliases, repeated returns, unused results and zero-input
nested bodies have independent controls. This is bounded static and already
audited B-floating composition, without general integer, complex or symbolic
BOOL product admission.

Placement requires actual unspecified sharding sentinels, None layouts and an
actual empty Mesh; donation and compiler options remain excluded. The
compatibility seam accepts the installed native `Inline.AUTO` and
`Inline.JAX_LATE` members on JAX 0.11 and real Boolean hints on JAX 0.10.
Foreign enums, matching strings, integers and unsupported native variants do
not pass through coercion or name matching. The existing protected JIT lowerer
freshens variables and reuses the shared validator/dispatcher. It introduces no
callable retrace, hidden specialization, lazy-body bypass or new cache system.

The eleventh changed path is a test helper in `test_source_finalization.py`.
After the locally audited LeakyRelu is removed from its partial validator view,
that view now declares the removed outputs as additional formals. The original
graph, constants, alpha overrides, negative controls and file/runtime cases stay
intact. Production binding validation is unchanged. All 54 finalization cases
pass independently on both stacks. Seven explicit custom-Where schema-corruption
controls and the existing full/dynamic-arange exclusions remain.

## Numerical evidence and unresolved raw gather

Strict Where zero-sign samples remain **FAILED** for FP16, FP32 and FP64, with
CPU optimization disabled and all, on both measured stacks: six failed cells
per stack. The permanent guard checks the recorded sign-only mismatch and that
`require_sample_acceptance` raises. A separate positive-zero control does not
replace the native reference or relax its bitwise criterion. A passing regression
that recognizes a numerical failure is not numerical acceptance.

The independent minimal audit found no corrupted artifact, reversed operand,
misbound reference or feed mismatch. Its direct model contains exactly one
standard ONNX `Where(true, x, y)` with a scalar true initializer and no arithmetic
or Cast. Native eager/JIT preserve `[-0, +0, -0, -1]`; the recorded outputs are
`[+0, +0, +0, -1]`. This is a measured target-execution discrepancy for these
artifacts on x86_64 Linux/WSL2 CPUExecutionProvider, ORT 1.31.0/JAX 0.11.2 and
ORT 1.29.0/JAX 0.10.2. It does not establish an ONNX representation impossibility,
a provider-wide exclusion or a general dtype/shape/GPU limit.

Ordinary exact finite and BOOL samples, NaN-class comparisons and captured
payload checks remain separate evidence. Existing finite Exp composition uses
its fixed FP32 `rtol=atol=1e-5` and FP64 `rtol=atol=1e-7` criteria. BF16 has
typed/schema/native evidence without a CPU-kernel acceptance claim. No new
five-row or twelve-row corpus numerical execution is released by this packet.

Both artifact-only gather supplements preserve the original source, actual
feeds, references, settings and history. Complete original and activated typed
root traces now agree. The raw x64-off scenario still stops before tracing at
its FP64/x64 configuration conflict. Raw x64-on advances past custom Where,
but fails the unchanged raw structural callback: it expects a scalar false
constant `()`, while native JAX explicitly broadcasts that constant to `(2,3)`.
Neither scenario gains accepted artifact success or an ORT call. Its original
None criterion is unchanged. The initial preparation attempt with mismatched
backend settings is retained separately from the replay under the frozen
historical CPU settings.

A native SOURCE counterpart for that gather row is being authored separately.
It has no accepted integration at this checkpoint and cannot waive or overwrite
the raw callback failure. The strict historical seventeen-source numerical CLI
also remains pinned: both negative checks pass because the CLI exits 2 on
changed provenance before preparation/ORT, writes no output and retains the
old approval and accepted artifacts.

## Shared x64 restoration dependency

The first full-suite failure exposed a pre-existing shared LEGACY context bug.
With a local x64 override active, `_force_jax_x64` saved the effective setting
and restored it into the hidden global slot. The separately reviewed two-file
repair saves the actual public `jax.enable_x64.get_global()` value and restores
it in `finally`, including when an entry-update hook writes then raises.
The body still uses the existing global update, so effective LEGACY behavior
under a local override is preserved. SOURCE keeps its existing no-update path.
Every original cache/state assertion and all eleven Where/JIT files stay exact.

Its new module adds 87 permanent cases: 24 global/local/request/body-exit
combinations, 48 nested combinations, one inner-local-override exception, twelve
entry-update failures and two calls of the original cache test under hidden
True/local False. The owner also runs both original pytest IDs unchanged, for
89 passes per stack (8.76 and 8.82 seconds). The earlier 77-case records and
direct-file mypy failure remain historical; final configured package typing
passes all 542 files. All 861 other existing Python files are unchanged.

Both post-repair main selections
pass 89 cases with hidden x64 True; five actual MAIN imports, all 863 Python
hashes and restored state are verified. Independent integration review accepts
these results.

## Validation and actual-main binding

| Selection | Python 3.12 / JAX 0.11.2 | Python 3.11 / JAX 0.10.2 |
| --- | --- | --- |
| Owner complete focused selection | 2,661 passed, 206 warnings, 191.97 s | 2,661 passed, 196 warnings, 204.26 s |
| Independent semantic review | 200 passed, 25.02 s | 200 passed, 25.52 s |
| Actual MAIN integration before x64 repair | 2,661 passed, 206 warnings, 437.14 s | 2,661 passed, 196 warnings, 310.59 s |
| Actual MAIN x64 restoration gate | 89 passed, 9.20 s | 89 passed, 9.10 s |

The packet adds **216 permanent cases**, including the three final AbstractRef
and high-form negatives; the earlier 213-case selection is not an additional
count. Independent review runs 29 new guards, 117 unchanged Pointwise cases and
all 54 finalization cases. All selections overlap and are not additive coverage.
The earlier Inline representation, fixture, empty-BOOL and partial-finalization
failures remain recorded. Final typed compatibility and narrowly corrected test
expectations retain the native source contract; no target failure is waived.

Both pre-repair final broad collections retain every case from the earlier 2,658 and
1,489 selections. Their exact 2,661-case digest is
`7ec926a2656ff90858c87a7cc966511ae7c072df37929945f6114656cfe1928b`.
Each has 7,983 passing setup/call/teardown phases, with no failures, errors or
skips. The owner records 550 observed production/support imports and pins the
49 selected test files. Actual MAIN execution additionally observes all 49 test
modules: 44 resolve to MAIN and five to the declared immutable external files.
Ten required production/support imports match MAIN before and after; all loaded
project/test modules match their declared MAIN or frozen external bindings.
Generated output directories alone move under `/tmp`.

The pre-repair actual integration starts from `58ff66e0ea1644a80f8952af96eeab714d059e0e` and the
accepted eleven paths. Independent review verifies every pre/postimage,
eleven protected dependencies, all 862 Python hashes, effective/hidden settings,
original JNP/NNX callables and the restored LEGACY policy. Both focused main
runs use one CPU and pytest `-n0`; their wrapper times are 438.219148 and
311.527023 seconds, distinct from pytest time above. Whole-package mypy passes
542 files, and the pinned owner Ruff, Black, IR-builder and diff checks pass.

The first complete primary-stack run records **7,147 passes and two failures**
(13,130 warnings, 407.62 pytest seconds). Both cache-order tests expose an
existing shared LEGACY x64 restoration defect: an effective local override was
restored into the hidden global slot. The failure record and all state assertions
are retained. The narrow shared-helper repair above has independent acceptance;
its fresh 89-case actual-main gate passes on both stacks. The failed first run
remains unchanged evidence of the defect; the fresh successful full run follows
the accepted repair.

The fresh complete primary-stack run passes **7,236 permanent tests**, with
13,130 warnings and no failures, errors or skips (400.95 pytest seconds;
402.217547 wrapper seconds). It includes the 216 Where/JIT additions and 87
restoration additions. This full run is distinct from the pre-repair dual-stack
2,661-case selection, which includes five external review files. The actual
post-repair secondary-stack gate is the fresh 89-case selection above; no new
secondary full-suite result is claimed. All fourteen checkpoint paths and all
863 Python hashes match before and after. Code commit
`ae820b1a357452d46046a984bf5cb060608c2143`, directly after `58ff66e0`,
passes all seven actual configured hooks without edits or bypass. All fourteen
checkpoint hashes and all 863 Python hashes remain exact after commit; the
worktree is clean.

The inventory changes only manual tests, manual provenance and counts. Where/JIT
adds 32 test functions in two new files; the shared x64 repair adds five functions
in one further file. The final inventory records **213 files, 1,254 functions and
502 migration additions**. The pre-repair 212/1,249/497 inventory and its CLI
records remain historical evidence. All 752 original anchors remain, with no
removal or rename, and the 1,802 raw rows/3,604 scenario cells stay unchanged.
Both refreshed actual inventory CLI comparisons pass; the secondary stack
retains its conditional `one_minus_square` baseline entry. Inventory identity
does not establish SOURCE acceptance for every row.

## Remaining work

The separate native gather counterpart and coherent typed operator families
are being authored. Native-reference preparation is not product admission, and
those follow-ups are neither accepted nor integrated here. Broader JIT and
Boolean boundaries, the 310-row symbolic inventory, seven NCHW rows, all 22
parameterized models with their actual runtime controls, 41 normalization rows,
104 metadata-free registrations and optional MaxText/MaxDiffusion discovery
remain part of the B–F completion contract.

The final interface must preserve existing fixed source implementations,
including supported runtime controls and FP64 work, through one normal export
contract. There is no permanent public precision selector or per-operation
LEGACY fallback. Both explicit `enable_double_precision=True` and `False` keep
their existing public behavior now; actionable migration errors belong only to
the reviewed final phase-F transition.

## Frozen evidence

Each record retains its own source binding and historical status. The owner
proof and early review predate actual MAIN validation; their pending later gates
are not the current status of this checkpoint. The artifact hashes below bind
the accepted implementation, failures and later validation separately.

| Evidence | Local artifact and SHA256 |
| --- | --- |
| Accepted eleven-path patch | `/tmp/jax2onnx-where-jit-eleven-files-v1.patch`, `9e7ac72a15ad717c808bbf11b3554da09c77f2586aa5441ff33d01d6cac3775f` |
| Exact code freeze and protected dependencies | `/tmp/jax2onnx-where-jit-code-freeze-v1.json`, `70e1e8b9ebb528964bf1fcd412fcf460ec4e522dce29edcd3f7bbe56fe0a65e8` |
| Final owner validation and preserved failures | `/tmp/jax2onnx-where-jit-final-proof-v1.json`, `ee4e0eeaf5848543bb20cbdc2ba83d5a848a7e83d785e63424cc3d41016fa9a1` |
| Independent semantic acceptance | `/tmp/jax2onnx-where-jit-astra-review.md`, `7136ea51890e560df26ca70568781ac04b9a83fde8710db91a6de1d72f4f865f` |
| Independent semantic verification | `/tmp/jax2onnx-where-jit-astra-verification.json`, `343ca04544bd770e19e4bf29ba6622c595d8876f161b513f5d9efdfd960aaa80` |
| Actual-main integration acceptance | `/tmp/jax2onnx-where-jit-main-integration-astra-review.md`, `997494862444672fd6ada856a7eb0c76932939907cd7efe3624dbf8c5c058ef0` |
| Independent actual-main verification | `/tmp/jax2onnx-where-jit-main-integration-astra-verification.json`, `17793e53e6c49fd253b36bbaf8fb37d05e52cf7a80ff3b07cbcdc14615286a50` |
| Both actual-main execution bindings | `/tmp/jax2onnx-where-jit-main-dual-verification.json`, `2fb93a7db4f210ed440208198d1da8f21ad286e24cb280b355dc9981283accde` |
| Independent strict signed-zero audit | `/tmp/jax2onnx-where-negative-zero-astra-audit.md`, `e697894cab76ba2937ff80f61be5d400c9f915411b56611ccf22dc6ae49d3f7e` |
| Signed-zero audit verification | `/tmp/jax2onnx-where-negative-zero-astra-verification.json`, `5ec9d3ca08ca09bdb48f8fb16c2d3cd8ba670e6f34241125c9f03b4c64ded4ca` |
| Unchanged raw gather: Python 3.12 | `/tmp/jax2onnx-where-jit-gather-py312-supplement.json`, `a71b88fd0f0a3c24a76bc261d3ded63a6eb5ad07e26e7e3c2babcb0b78ed5a4d` |
| Unchanged raw gather: Python 3.11 | `/tmp/jax2onnx-where-jit-gather-py311-supplement.json`, `9512d5df9f7eb698c1c2b077751eb12fc65e70b2653c8d65078d7cc438ab4033` |
| Pre-repair inventory | `/tmp/jax2onnx-where-jit-inventory-candidate.json`, `14c13d7a856c1fad63eeff0312bc2e4a9c2abda0665ab8885dee5e0585feb00b` |
| Pre-repair inventory CLI: Python 3.12 | `/tmp/jax2onnx-where-jit-inventory-cli-py312.json`, `830812338e215b094d9436cd36407fca0e421408e52f06e0c7603fceb4699479` |
| Pre-repair inventory CLI: Python 3.11 | `/tmp/jax2onnx-where-jit-inventory-cli-py311.json`, `ced8fd68f343e83d7d63301b43da0ebe570057da706396433c64268f296673eb` |
| First failed complete primary run | `/tmp/jax2onnx-where-jit-main-full-result.json`, `43538830fc4a94904f1f699e82bbfe5dbebe69aa7c630576b923d2abc17d4bb7` |
| Independent original failure diagnosis | `/tmp/jax2onnx-where-jit-full-failure-astra-audit.md`, `f8f49c8e9ea3058b63f6ea107e8e7f552ba0b47de7f9511c4ea5a3c826e77854` |
| Exact shared x64 repair and owner evidence | `/tmp/jax2onnx-x64-global-freeze-v1.json`, `0b16577e7b327fbde608ce48807a37cf40ec3d172c26579d99c914381b332e1e` |
| Independent shared x64 repair acceptance | `/tmp/jax2onnx-x64-global-astra-review.md`, `5286eb3b6c031884ec8ef1852a1b0b8ef0425b5d1501b1a350c28f9724d34e2c` |
| Independent x64 repair verification | `/tmp/jax2onnx-x64-global-astra-verification.json`, `a53332815bd985b5b902569df2f1787de10f3fd21e8b10b6a3619b2094d3fc31` |
| Both actual-main x64 repair bindings | `/tmp/jax2onnx-x64-global-main-dual-verification.json`, `dc35e4145ef6e6af2d1c2e6b241bcdf17248993270ef784a56c45153cde5ac9b` |
| Independent x64 actual-main acceptance | `/tmp/jax2onnx-x64-global-main-astra-review.md`, `8d34bc15468637a61e099eb764dfc7705ae4ef7ed935b09f6685351fc42098a5` |
| Independent x64 actual-main verification | `/tmp/jax2onnx-x64-global-main-astra-verification.json`, `47196ecffca1da2440454a0455d2825558df5535141011acc4c0b48a22a6f583` |
| Final inventory including shared x64 repair | `/tmp/jax2onnx-where-jit-inventory-candidate-v2.json`, `ce4ac9a85b134f37dc9695dbd35f0ea838d927db158bb42ce0a6fb61f7c42266` |
| Both refreshed inventory CLI checks | `/tmp/jax2onnx-where-jit-inventory-v2-dual-verification.json`, `d0de97ac128ce1e0078cce57450fd70e8601a7565562cd664283aa1503cbde4e` |
| Exact five added restoration test functions | `/tmp/jax2onnx-where-jit-inventory-v2-check.json`, `4e0778cfe8828b1da003e8918a3d23d6249ed9e415e4a07edee8f355c2d619b2` |
| Fresh full-run input freeze: fourteen paths and 863 Python files | `/tmp/jax2onnx-where-jit-main-full-freeze-v2.json`, `aa294f6ad8806e5ec7ca1ae32b4a20ff1d60678c9a4e2f7f1f9ed35143fe29c9` |
| Fresh complete primary run after shared repair | `/tmp/jax2onnx-where-jit-main-full-result-v2.json`, `a5d37ddb18de77ab18200ca03f0c576b495da8b9ce061cce322856ca86eb573a` |
| Committed code checkpoint and unchanged source bindings | `/tmp/jax2onnx-where-jit-main-code-checkpoint.json`, `dce57512114380f9c655152b5074f477d7dd533529bf87b276fae90be253a4aa` |
| Actual seven passing code-commit hooks | `/tmp/jax2onnx-where-jit-main-code-commit.log`, `dcc5f44b7051b367de2a8bcadcc0034dccdba268aa6ab8c8a4d24e32b5938944` |
