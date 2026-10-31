# Typed values and masks: bounded private checkpoint

Status: committed private checkpoint, `0d8f9d194d11e4ce499ac03df3abbc48dc1552ee`,
with independent semantic, owner and actual-main acceptance. The complete
primary-stack suite passes 8,659 tests; both inventory CLI checks and all seven
actual commit hooks pass. Earlier failures and the precise execution epochs
remain recorded below.
This is an internal SOURCE checkpoint. Public LEGACY behavior, both existing
`enable_double_precision` options, profile `source-static-v1` and report schema 1
retain their meanings. Full existing functionality through phases B–F remains
unfinished.

## Representation and admission

The checkpoint adds coherent rules for BOOL and signed/unsigned integers of
8/16/32/64 bits alongside the existing floating types. It preserves native JAX
promotion, weakness, literal attributes, actual SymbolicScope identity and
formal/captured bindings. Comparisons, bitwise operations, explicit conversions
and two-case selection follow their actual typed contracts. Integer selection
uses exact local integer carriers, including UINT64 values across the signed
boundary, without a floating-point round trip. Canonical operands and sibling
consumers remain unchanged.

The finite symbolic rule still requires one native leading B at rank two or
three with matching positive static tails. Pointwise binary operands must have
the same B and tails, or the supported static leading-one shape. Selection keeps
equal arm/result signatures and a scalar BOOL or matching same-scope mask.
Ordinary static broadcasting is separately checked. The original implicit
rank-zero/B composition remains a temporary implementation-gap negative; an
explicit singleton/broadcast positive does not replace that original source.
General integer arithmetic, rank-one B, arbitrary symbolic broadcasting and
broader scope admission remain open.

Complex64/complex128 support here is a static foundation: typed boundaries,
construction and real/imaginary extraction with the declared physical component
axis. Complex arithmetic, comparisons, conversions, selection and symbolic
complex scopes are separate work. Native component broadcasting is preserved by
SOURCE-only local Expand values before the unchanged packing helper. Existing
matching components are reused; captured payloads, original operands and sibling
outputs remain intact. LEGACY behavior and the shared packing helper are unchanged.

The reviewed implementation spans 21 production paths and ten test paths,
including three new test modules, against 867 Python files. The root inventory
is a separate thirty-second checkpoint path. Its profile/admission rules do
not become a public mode. A successful type or schema check does not establish
every operation, target kernel or numerical value in that family.

## Review correction and permanent controls

The first independent review rejected a real lowering defect: native FP32
components `(2,1)` and `(1,3)` produce complex64 `(2,3)` and pass the proposed
profile, but the old packing call rejects their unequal shapes. Both supported
stacks reproduce that failure before ORT. The local Expand repair resolves the
admitted case without inventing a new exclusion or weakening the protected helper.

The v3 repair review accepts the source change. Its focused predecessor run
passes 35 cases per stack. The v4 test-only extension adds reverse array/scalar
coverage, giving 32 positive complex cases and two native unequal-rank negatives.
The initially mistaken `(3,)`/`(2,1)` positive is retained as a native
trace/eager/JIT and SOURCE-error control; native JAX rejects that unequal nonzero
rank pair. The original failure records remain unchanged.

The three new modules collect 1,330 cases, including the original 650 native
cases, eight UINT64 capture/sibling cases and the complex additions. The v4
primary selection passes 1,330 cases in 179.89 seconds with 32 warnings and
restored state/source hashes. That is a predecessor result, not final broad or
actual-main acceptance. Earlier dual 1,288-case results likewise remain historical.
The 650-case preservation supplement proves matching successful IDs and their
inclusion in the later selection, plus recorded current-prefix/criterion facts.
The original complete source/header preimage is unavailable; no historical
whole-file byte or whole-AST identity is claimed.

The first broader selection collected 4,170 cases and stopped after 320 passes
at an obsolete integer-Dot rejection location. Typed inputs now pass the input
gate; the unsupported integer equation correctly fails later. The authorized v5
change updates only two expected-location literals for integer/complex matrix
negatives, preserving their rejected models, criteria and production rules.
The replacement map retains 2,653 unaffected prior cases and maps eight old cases
to ten, for 2,663 prior cases; with 87 x64-restoration, ninety Gather and 1,330
new-module cases, that broad selection was 4,170. The subsequent v5 primary run
reported **4,162 passes and eight failures**; there was no secondary v5 run.
A rejected adaptation proposal incorrectly grouped Sub with the constant-True
Cond positive. The accepted v6 correction preserves the original Sub and ReduceSum
negatives, moves only that original Cond to the native-bound B=2/B=5 positive,
and retains the Add B=3/B=1 mismatch negative. BOOL/UINT32 identity/options and
integer/BOOL Exp controls retain their separately reviewed schema/facade bounds.
No broader non-Add symbolic admission is inferred.

The final map replaces seventeen old IDs with twenty authorized IDs, retaining
all other original identities and yielding **2,664 mapped baseline cases**.
Both final v6 owner runs collect and pass **4,171 cases** across all 54 roots,
with zero deselection, failures, errors or skips and 12,513 successful test
phases per stack. Primary records 136 warnings/462.44 seconds; secondary records
125 warnings/616.92 seconds. Independent post-run review accepts this closed
owner evidence. All original 650 native IDs are present; the new modules account
for 861 predicate, 262 conversion and 207 boundary cases, totaling 1,330. The
650 native cases are included in that total. Gather's ninety and the x64 repair's
87 cases are retained. These overlapping descriptions are not additional passes.

Production and all three new modules remain unchanged from v4 through v6;
only the reviewed matrix, pointwise and symbolic-batch test files differ. The
original v5 failures, rejected proposal and earlier source/header provenance
limits remain recorded. Passing negative or failure-recognition tests does not
admit their rejected programs or erase numerical failures.

## Original 65-row observations

The frozen selection contains sixty core rows and five adjacent rows. Both x64
settings are retained for every unchanged callable, yielding 130 raw cells per
stack. Independent review accepts the following bounded owner observations:

| Per stack | Result |
| --- | ---: |
| Core cells / audited artifacts | 120 / 119 |
| Adjacent cells / audited artifacts | 10 / 6 |
| Total cells / audited artifacts | 130 / 125 |
| Retained gaps | 5 |
| Separate validated bitwise sample records | 500 |

The five gaps are explicit:

- Raw 188, `plugin::broadcast_in_dim`, ordinal 3, both x64 settings: original
  `broadcast_in_dim_batch` reaches the protected custom `jax.numpy.zeros`
  forwarding/admission gap. The diagnostic also retains its unsupported
  symbolic-shape parameter serialization marker. B=2/B=5 native preparations
  exist, but there is no accepted artifact or runtime result for those variants.
- Raw 189, the same registration, ordinal 4, both settings: original
  `broadcast_in_dim_dynamic_B` fails the rank-one `[B]` preparation boundary.
- Raw 205, `plugin::complex`, ordinal 2, x64 off: its explicit FP64 input would
  canonicalize to FP32. The configuration conflict remains; x64 on succeeds.

These are configuration conflicts or temporary implementation gaps, not proof
that ONNX cannot represent the sources and not waived requirements. Of the 125 admitted
cells per stack, 98 original callbacks pass and 27 have no callback; absence is
not callback validation. Saved export reports retain runtime/numerical
**UNVERIFIED**, with admitted representation/lowering checks **VALIDATED**.
The raw None criterion and legacy tolerance fields are unchanged.

The separate numerical lane freezes bitwise criteria and original native
eager/JIT references before target execution. Each artifact has four separate
sample validations: eager and JIT references under disabled and all CPU
optimization. All 500 records per stack validate. Review verifies 256 file pins,
250 saved models with the full ONNX checker, and 1,000 reference/output-hash
bindings across both stacks. It also checks retained native payloads, original
input contracts and settings.

The 500 target executions per stack are supported by the frozen helper's one
`session.run` per successful one-sample report, with no retry or warmup path.
There is no separate session spy. These are separate executions, not one saved Y
reused for both references. Actual Y payloads are not retained; typed output
hashes are. No independent saved-Y replay is claimed. Targets are the recorded
CPUExecutionProvider combinations: JAX 0.11.2/ORT 1.31.0 and JAX 0.10.2/ORT 1.29.0,
sequential execution and one intra/inter thread. Source hashes, caller settings,
policy and watched native callables are restored. The raw None lane performs no
ORT call. The first tuple/list contract-comparison failure is retained; v2 fixes
only diagnostic JSON normalization, without changing source or criteria.

## Cast references and retained numerical failures

Fresh persisted native replay drivers reproduce the 123 frozen cast contracts
on both stacks. The secondary driver explicitly uses the primary expected
receipt; root verification confirms that the original byte-level contracts
agree between stacks for all 123 cases. This new replay does not retrospectively supply the
unpersisted original driver. The earlier NumPy-host conversion discrepancy
remains a strict failure: the prepared strong-JAX-array reference is distinct
from the observed primary-stack host fast path. Do not claim universal native
eager/JIT agreement across argument representations.

Float-to-integer lowering uses the reviewed local saturation/NaN treatment.
Ordinary floating conversion retains plain Cast. The separate durable subnormal
packet records ten serialized artifacts per stack, twenty direct calls with saved
Y bytes and forty additional validator executions: sixty actual target calls per
stack. Comparing the twenty saved outputs against both native references gives
28 exact matches and twelve strict mismatches per stack. Separate validator
statuses agree, and all twelve failed gates raise. Validator output payloads were
not retained, so they are not the same recorded executions as the saved Y values.

The mismatches remain the original FP32→FP64 signed-zero group and the
FP32→FP64/FP64→FP32 boundary groups on the measured CPUs. No tolerance is loosened,
subnormal domain removed, or FTZ adapter added. MXCSR and a causal denormal mode
remain unobserved. These findings do not establish a portable flush policy,
provider-wide exclusion or ONNX impossibility. Passing failure-recognition tests
does not turn those numerical samples into accepted support. The older Where
negative-zero failures and BF16 runtime limits remain separate and unchanged.

## Original-model boundary progress

New actual receipts, distinct from the earlier SIDE-normalization record, check
the unchanged GPT and GPT_Embeddings registration recipes with their original
INT32 `[B,1024]` inputs and `input_params` identities. Both x64 settings pass
input normalization and validation on both stacks. The unchanged SOURCE option
guard then rejects `input_params` before factory invocation, weight construction
or forward execution. No complete native model trace, model artifact or ORT
result is claimed. The first driver used incorrect shortened registration IDs;
its failure is preserved, and v2 corrects only those provenance IDs.

The original Gather row retains the x64-off FP64 conflict. With x64 on, its
native reference proceeds and SOURCE now stops at integer index addition outside
the profile. This is a new measured first barrier, not a replacement for the
historical [Gather diagnostic checkpoint](gather_counterpart_findings.md), its
raw callback failure or its None criteria. Both model-progress receipts record
zero ORT attempts. They do not accept the full 22-model compatibility set.

## Native-reference context

A separate frozen investigation of `plugin::dot_general` ordinal 6 and
`plugin::jax.numpy.matmul` ordinals 2/4/8 reproduces all earlier owner/main
reference-hash differences on both stacks and x64 settings. With the same inputs,
callables, typed traces and StableHLO, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`
matches MAIN, while an empty value matches OWNER. This proves a sufficient
context dependency in the controlled experiment. The historical owner's
XLA_FLAGS value was not recorded, so historical attribution remains unproved.

The largest observed owner/main difference is 69 ULP and approximately
`2.98e-8` absolute. Those distances are descriptive, not an acceptance tolerance.
No ONNX target ran, no criterion changed and no historical hash was repinned.
Independent review accepts this finite controlled observation, verifying 689
closed artifact pins, sixteen protected historical bindings, the 864-file source
map, 32 saved feeds and 496 saved native outputs. Ten observation jobs contain
eighty cells; that is not a count of eighty individual function calls. Equal
typed traces/StableHLO and restored environment/configuration are checked.
No emitted arithmetic-instruction schedule is proved, and the tested empty
XLA_FLAGS value does not establish equivalence with an unset variable.

The unchanged Gather findings retain their original unresolved historical
observation. Relevant native backend/thread settings must accompany future
reference hashes; equal source settings alone do not establish universally
identical native bytes. This acceptance selects no new golden hash and changes
no product behavior or numerical criterion.

## Validation and integration gates

Independent review accepts the actual MAIN integration and the later header-only
amendment. The validation sequence is explicit:

| Actual execution | Files executed | Passed | Warnings | Pytest seconds |
| --- | --- | ---: | ---: | ---: |
| Owner primary broad | v6, before formatting | 4,171 | 136 | 462.44 |
| Owner secondary broad | v6, before formatting | 4,171 | 125 | 616.92 |
| MAIN primary focused | v6, before formatting | 2,047 | 32 | 287.47 |
| MAIN secondary focused | v6, before formatting | 2,047 | 31 | 349.84 |
| MAIN complete primary suite | v7, after formatting and before path comments | 8,659 | 13,162 | 488.80 |

All runs exit zero. The MAIN focused selections have identical identities,
6,141 passing phases per stack, twelve observed test modules and seven recorded
actual import paths. Source hashes, checkpoint paths and captured settings,
callables and policy are preserved. Those checks are distinct from the owner
54-root selection and are not relabeled as postformat runs.

The v7 amendment changes only whitespace in `precision.py`, from
`afb7cc075e83…` to `4d69124a165e…`. Tokens and the AST, including type comments
after normalization of line-position metadata, are unchanged; all other 866
Python files stay exact. Repeated Black/Ruff checks pass all 31 code/test paths.
Configured mypy over 542 files and the original RNG checks passed on v6 before
formatting; those results are not new v7 executions.

The fresh v7 full run has zero failures, errors or skips. Independent review
reconstructs exactly its 8,659 unique identities from all previous 7,326 full-suite
identities, the authorized seventeen-old-to-twenty-new mapping, and 1,330 new-module
identities. There are no missing, unexpected or duplicate identities. The run
preserves all 867 Python hashes and 32 checkpoint paths. Its 488.80 pytest seconds
are distinct from the 490.3189-second wrapper duration.

The first actual commit attempt fails only `ensure-path-markers`; the other six
hooks pass. V8 adds exactly the repository-path comment and a blank line to each
of the three new test modules. Their entire prior bytes remain exact suffixes,
and full ASTs including type comments are unchanged. The other 864 files,
including all 21 production paths, stay byte-identical. Independent review accepts
that amendment. The full suite remains a v7 observation, before those comments;
it is not claimed as a postheader rerun.

The actual retry passes all seven installed hooks in 49.1307 wrapper seconds and
creates local code commit `0d8f9d194d11e4ce499ac03df3abbc48dc1552ee`, parent
`8d08edea24aea808f8b9d07fc7e526a68d5231d9`. All 32 final checkpoint paths and
867 Python hashes remain unchanged after commit, with a clean worktree. No hook
is bypassed; the original failed attempt remains pinned.

The inventory records **217 files, 1,323 functions and 571 migration additions**.
All 752 original function anchors, zero original removals, 1,802 raw rows and
3,604 cells are retained. Only manual entries, their provenance and the two
corresponding counts change. Seven reviewed migration-function replacements
and 66 new names give a net 59 functions; they do not remove original anchors.
Both actual inventory CLI checks exit zero with source/checkpoint hashes preserved.
Inventory and regression success do not turn the retained raw gaps, numerical
failures or unfinished model contracts into support.

## Next shared work

Independent planning review accepts the shared-parameter proposal and the
precision-preparation decision **with release conditions**. Root has released the
named-parameter packet for isolated implementation against the committed Typed
checkpoint; independent review, integration and model acceptance remain pending.
The release contains exactly six paths: `conversion_api.py`, `user_interface.py`,
existing `source_precision_support.py` and `test_source_runtime_parameters.py`,
plus new `test_source_named_parameter_facade.py` and
`test_source_fixed_parameter_models.py`. Keep profile, precision helper, IR
context, plugin system, lowerers and original model/registration files pinned;
a concrete additional dependency needs separate ownership review.

The released isolated implementation prioritizes the existing named-runtime-input
mechanism through the audited facade and the same
source-preparation/reference mapping. Preserve original declared types,
signatures, arity/defaults, weakness, actual scope identity, names, reports,
serialization and cache behavior. Runtime flags stay dynamic through the whole
model; static binding is only for an intentionally fixed setting. Existing
external whole-root adapters must preserve complete unchanged source behavior,
including randomness and state prerequisites.

The first complete-model target is unchanged NNX MLP ordinal 2, followed by
Linen MLP and Function examples 014/015. Retain actual weights and positive
Dropout rates. Test both flag values and applicable B variants through the same
artifact, with prefrozen discrete/intermediate and floating criteria or explicit
None/unverified status. Keep all 22 original identities and actual remaining
normalization, activation, randomness and other model barriers. Input-parameter
support alone will not complete those models. The six-path packet is released
for isolated implementation; independent review, integration and model acceptance
remain pending.

RNG and state obligations are model-specific. The frozen prerequisites distinguish
NNX training that advances a stream from inference that consumes none, the actual
Linen bridge's repeated fixed-key behavior, Equinox's original missing-key error,
and SpectralNorm's supported fixed `update_stats=False` contract. Preserve required
fixed-key dependence, split/advance relationships, shared-key correlations,
repeated-call behavior and observable state updates. Where exact key-to-output
behavior is required, it remains a requirement. This is not a new universal
promise of bitwise JAX random streams across runtimes.

A live `Not → Dropout` graph or a distribution test does not automatically prove
training equivalence. Patched Dropout eager implementations that return their
input are not original native references. Compare native eager and valid JIT
from the same declared initial state and bind the resulting post-state, using
appropriate framework state handling. Do not compare successive calls that
consumed different streams as identical inputs. Keep numerical criteria None
where the actual model contract is unsettled; a deterministic binder can be
correct while that model's training acceptance remains open. Do not invent new
state/key ports or a universal RNG engine inside this six-path packet.

Fixed-source precision preparation is required compatibility work before the
final transition. Its one placement is on the JAX side before ordinary export:
existing configuration where available, otherwise an explicitly specified and
validated prepared computation, with unchanged supplied source files and a
matching native reference. Deliberate FP16/BF16/FP32 casts, required accumulation,
stored values and output contracts remain protected unless an explicit numerical
redesign changes them. Wider parameter storage cannot recover lost precision.
A general transformer is not implemented. There is no outstanding general scope
question about whether fixed sources must work; individual numerical changes
still require explicit intent. This does not create a public precision mode or
per-operation fallback. Both explicit legacy Boolean values error only at the
final reviewed phase-F transition.

## Frozen evidence

The temporary evidence below is pinned for review, not a maintained public API.
Older findings and raw records remain unchanged. The integration, full-suite,
header-amendment and actual commit receipts bind their distinct execution epochs.

| Artifact | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-typed-values-post-gather-authoring-freeze-v4.json` | `556e55f6ff75bbf1f5dbe83d98d67b6334b7b8a33f886cbfc896376c29126695` |
| `/tmp/jax2onnx-typed-values-review-snapshot-v4-manifest.json` | `a216ebca2103585a05a9f1d76771dc388ca0ed6f4dbaf0f7584023ab3b2801b8` |
| `/tmp/jax2onnx-typed-values-review-snapshot-v5-manifest.json` | `d8f2dc8a309e3f7f465969e25852d332744b38e43deb8f38c860afa69f45997e` |
| `/tmp/jax2onnx-typed-values-obsolete-control-replacements-v2.json` | `0292ed1b5a8729a0a952afe5d2ee12af8194e615ec0e388fe015e359ecfa7f9c` |
| `/tmp/jax2onnx-typed-values-semantic-astra-review-v1.md` | `e88d3b80bb90bb24cd2da087e0ed574abca0ca2237c9d91dda17ea321d58e83a` |
| `/tmp/jax2onnx-typed-values-semantic-astra-review-v3.md` | `4bd0d6547bd5ba01e2ff9faf4001c2badaba8e3a7894194ac60d8e9dff26b994` |
| `/tmp/jax2onnx-typed-values-semantic-astra-v4-supplement.md` | `3b1c524df2c68fd743c0f0cda49a9c973a1454e81f28426968e4a37864c423d8` |
| `/tmp/jax2onnx-typed-complex-broadcast-diagnostic-v2-manifest.json` | `e593b73de5d6b94336282eac268101ab3b1e78b9c527e8c8f5760fad27f71cb8` |
| `/tmp/jax2onnx-typed-values-native123-dual-root-verification.json` | `b690aba79173d7a13f450485ef0a777180232fbf56e6e07cd36705bda3aa211f` |
| `/tmp/jax2onnx-typed-native650-preservation-supplement-v1-manifest.json` | `b0bd2f8c74465676160d6ef87ba34061240603ed45e1d33ad4186d9c58971303` |
| `/tmp/jax2onnx-typed-raw65-dual-v2-manifest.json` | `45919713a3f0d51645dfae5f696b10f697cece6156031b2a625923f67b0a89d3` |
| `/tmp/jax2onnx-typed-values-raw65-postrun-astra-review-v1.md` | `5bae7ca6d35b8185a91224fae9d0031e927db67635d4d3a276ff3f6ae592799b` |
| `/tmp/jax2onnx-typed-values-raw65-postrun-astra-verification-v1.json` | `daca57c8e551d9845afe7b03a64e98736945ff53e056198182ae113db5d2ef5c` |
| `/tmp/jax2onnx-typed-original-model-progress-dual-v2-manifest.json` | `2697599dffcb3276cdc85f4dda710d1a54d24c8988c6e469293cbeea8c6f8a98` |
| `/tmp/jax2onnx-typed-cast-runtime-evidence-manifest.json` | `0ec856d58eeb2a9310f3bc7fd69f1ee9a3352608a677e133d33da3c752cc5ffc` |
| `/tmp/jax2onnx-typed-cast-runtime-execution-accounting-clarification.json` | `ab2e9a06e1d2abb59a6ab83bc7c921974dd517e89793c4d5abbd65cf1a5d4dfb` |
| `/tmp/jax2onnx-typed-cast-runtime-astra-supplement.md` | `c7177fef2f08a54ab1f1a4f35c557b22ca45d2c49f000fa46113c5dd287bc904` |
| `/tmp/jax2onnx-native-matrix-reference-findings.md` | `e8a51a2cde4acf4fa13f5f913d97c77cd2db77c9a0dcebc6ab6c55f2313141c2` |
| `/tmp/jax2onnx-native-matrix-reference-manifest-v2.json` | `dded8c4d708e0318917c08875073816cf25febb1a3e85dfa9895114cebb7061e` |
| `/tmp/jax2onnx-next-shared-parameters-plan.md` | `79f5a4727f17c9401d1066f6105adf639bae2053071e47ae2e125290714f013b` |
| `/tmp/jax2onnx-next-shared-parameters-bindings.json` | `634048191a92208f953c4ca7de30659fe5d221bbf00c0cd1aab19f7866540d51` |
| `/tmp/jax2onnx-fixed-source-precision-preparation-decision.md` | `2c8a6596186cb9a8151d22740201966a0617912d3a0220121640c5f43ef19779` |
| `/tmp/jax2onnx-fixed-source-precision-preparation-bindings.json` | `4fd445b755c939dc3d3fa8697f9b54f4178913311b53ed2d92967f3cd9cc1cc1` |
| `/tmp/jax2onnx-typed-matrix-boundary-attribution-release.json` | `a7ac87ab43ef6891672bf92c84ab5fac0df1c7da226e0043d22db5c666c25243` |
| `/tmp/jax2onnx-native-matrix-reference-astra-review.md` | `ebd587732f1bdb5c31d236fb167d83722a4d02ca2771e8427a8ca91b9092d350` |
| `/tmp/jax2onnx-native-matrix-reference-astra-verification.json` | `04fd1884282e4471cac363ad990af62f6a8cceccb42c1ded81b6c1b47e071a0c` |
| `/tmp/jax2onnx-next-migration-design-astra-review.md` | `6723091f6c5ca6a86f6b55626a614d136318f006e174f61cd02390b17213795b` |
| `/tmp/jax2onnx-next-migration-design-astra-verification.json` | `711afa9fdfb3d933e7488f2e099fbcdae32aeb4dc499658488179a2820839719` |
| `/tmp/jax2onnx-typed-values-owner-postrun-astra-review-v6.md` | `32b976b91c6d644b91a7295cda16867c5caf68add42b6960fb808f4f01a4d45a` |
| `/tmp/jax2onnx-typed-values-owner-postrun-astra-verification-v6.json` | `82fe12a375fd87e20a419fa02d893c6bbc0a0f76bd78ad8849240b3674feaffd` |
| `/tmp/jax2onnx-typed-broad-dual-v6-manifest.json` | `97e5182ca00c0e2db8e775d27f0a2cba746e5eae7f7aa2df72d8f9a1c02f1c05` |
| `/tmp/jax2onnx-typed-final-style-v7/execution.json` | `ba4843b36def0142f020465ad150b8ac845a705db18b02a7068acfa3f0b4a91c` |
| `/tmp/jax2onnx-typed-main-focused-dual-v1.json` | `b9b3e65d404c77d1eae6f43fec9a1a182fba22be29d5f0598952e8a7fe2c94aa` |
| `/tmp/jax2onnx-typed-main-inventory-verification-v1.json` | `a960bccf704375e306b5190fe95b1c55d30fb0c11cded7d9cd45f803b25d3707` |
| `/tmp/jax2onnx-typed-main-inventory-cli-dual-v1.json` | `a0002b9d5d903f061ba3f2094dad0a02ccd5c08100945574d3e4283a3c22ebe2` |
| `/tmp/jax2onnx-typed-main-full-freeze-v2.json` | `0ff01aa63b61782c85bb380f4dc831fe7d2d172ec5bb3bd90b2402d726f4ebb6` |
| `/tmp/jax2onnx-typed-main-code-checkpoint.json` | `1fbea58dbcf771cc84b31e8458da0d7ce809ec4a346807904dc3290b31be0488` |
| `/tmp/jax2onnx-typed-main-initial-hook-failure-v1.json` | `5edf1860787263eace772b65dc51e7670257a2b8d6c654c45c9911e27085a05f` |
| `/tmp/jax2onnx-typed-main-final-freeze-v3.json` | `5fe8675349fde7034b4c1a35ee6c39b0e818fb15bf2081a8337fc495aaa3125e` |
| `/tmp/jax2onnx-typed-main-header-amendment-astra-review.md` | `b6450e8d14877088837cb2f8281af2ef819346ffce002bc6f2a0fdef8aae0dc0` |
| `/tmp/jax2onnx-typed-main-header-amendment-astra-verification.json` | `1eec736905a9b86e14cd6d18303576aa98c9088fcfc29db9d12a7b9710cf460f` |
| `/tmp/jax2onnx-typed-main-integration-astra-review.md` | `2160069572e4f91abf02cc69d0db57608d8432d830847be611d03b3bdf169f94` |
| `/tmp/jax2onnx-typed-main-integration-astra-verification.json` | `76ad71168367088766d569c476be2a98e8f1d946d7057dd0f17f05cf77585859` |
| `/tmp/jax2onnx-typed-main-full-result-v2.json` | `ef5592dedb69eb866bba20a9566510f29ce91ee20ae7a68c47c93a9fa882828d` |
| `/tmp/jax2onnx-typed-main-full-identity-preservation-v1.json` | `d4bb2047399e434ae6e249a796cd5bcb090b23917ca53ede17b4d8c275cdaaec` |
| `/tmp/jax2onnx-typed-main-code-commit-retry-v1.log` | `a6738e5d5421376cd3d60b9b289f1999f1cccf7c18635b2ccee4a7e32ccea69f` |
