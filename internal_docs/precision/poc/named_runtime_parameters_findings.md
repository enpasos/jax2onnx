# Named Runtime Parameters: Audited Binding and Model Progress

The reviewed private SOURCE binder treats declared named values as real typed
arguments of the complete root trace. It removes the blanket `input_params`
exclusion without changing public LEGACY behavior, adding a user-visible mode,
or admitting every operation inside a model. The finite binder, sample results
and all-22 progress ledger are separate evidence scopes.

## What changed

The nine-path packet changes shared conversion binding, the facade, static
physical sample checks, the existing preparation helper, three existing test
modules and two new modules. Positional arguments precede named insertion order.
The actual native call supplies Python signature/default/error behavior; declared
runtime exemplars do not become frozen keyword constants. Typed shapes, dtypes,
weakness, symbolic scopes and actual Jaxpr variable identities remain checked.
Exported names, aliases, reports and physical feeds use the same mapping.

The facade still uses the default audited SOURCE validator. Named parameters
do not bypass primitive, precision, shape or scope admission. The additional
static sample check validates the actual physical ONNX input count, order, name,
type and extent, including complex channel storage, scalar and empty shapes.
Invalid feeds fail before optional ORT creation, or before a later run in an
existing session. No input cast, broadcast, tolerance or reference-policy change
is introduced. Diagnostic and metadata-free legacy paths retain their meanings.

The preparation helper traces and evaluates the original callable with the real
positional and named values under its declared scenario settings. Supplied
preparations are checked for original arity/type/weakness/shared-B compatibility
before payload identity, all before export. Named declaration and payload checks
also use the scenario's x64/matmul context rather than the restored caller state.

## Finite runtime evidence

The final owner affected-area suite passes **1,026 cases on each stack**:
primary 68.01 pytest seconds, secondary 62.34. This is an affected-area selection,
not a rerun of every earlier SOURCE suite. Collection preserves 151 old cases
injectively, including fourteen renamed IDs, plus 96 genuinely new cases: 247
across the three old and two new modules. Model-ledger cases have their own run.

The runtime observers cover root, ONNX Function and If examples, each using the
same artifact at B=2 and B=5 with True/False/True. Inputs are positional FP32
`[B,3]`, named FP32 `[B,3]` and scalar BOOL. CPU disabled/all use sequential
execution with one intra/inter-op thread. Native eager/JIT equality and the
bitwise criterion are established before export; the native archive stores their
common reference rather than separate eager/JIT arrays.

| Per stack | Sessions | Actual runs | Recorded evidence |
| --- | ---: | ---: | --- |
| Validation API | 6 | 36 | Six reports, six samples each; observed actual API outputs and recomputed input/reference/output hashes. |
| Separate direct capture | 6 | 36 | Separately created sessions and saved actual outputs compared to the same frozen references. |
| Combined observer | 12 | 72 | Three scope examples under two optimization settings. |

Across both stacks this is **144 actual runs**, including 72 API runs and 72
additional direct runs. It is not 144 validation-API calls. Independent review
checks all observed feed/output payloads and the 72 API sample identities.
The direct `actual-*.npz` archives and observed API output archives remain
separate. These are finite FP32/BOOL samples, not general FP64, GPU, RNG or
whole-model numerical acceptance.

## Unchanged models: progress, not acceptance

The closed same-v2 run records the exact 22 original rows plus one static
Function014 guard on each stack: 23 checks per stack, both exit 0. Model inputs
are zero tensors, symbolic B is concretized to 1, and named values retain their
original exemplars. All 22 criteria stay None; runtime, numerical acceptance
and product support remain UNVERIFIED, with no model ORT run.

The original NNX MLP, ordinal 2, reaches `nnx.batch_norm` at equation 3 after
its first Linear/bias path. Linen MLP reaches `linen.dense` at equation 0;
Functions014/015 retain LayerNorm barriers. GPT full and embeddings now reach
zero-rate `nnx.dropout`, beyond the old named-input guard and earlier iota
observation. Other rows retain their actual operator or rank-four symbolic-input
barriers. Original Equinox inference=False missing-key and SpectralNorm
update_stats=True immutable-state errors remain visible. Passing the ledger
means recording these outcomes faithfully, not accepting those models.

## Next model and exact selection work

The reviewed next model plan is released from accepted code `8f823ac3` for
isolated implementation, initially covering eight existing production paths and
three new tests. The required measured PRNG extension and floating-criterion
approval remain part of closure. It follows unchanged NNX/Linen MLP dependencies through
BatchNorm, Dense, GELU and Dropout, retaining actual weights, rates, flags and
state. Running-stat inference and native GELU approximation need separate checks
from training. An explicit pure whole-model state/RNG wrapper and the selected
PRNG extension are mandatory for both-control completion. Preserve key/counter
advancement, shared-key relationships, inference consumption, repeat and error
semantics. Distribution checks or a live Dropout node alone are insufficient;
no universal bitwise random-stream guarantee is invented. Nonzero B=2/5 probes
and criteria must be frozen before targets, alongside original zero probes.
The static state/PRNG supplement specifies branch-local reconstruction from the
actual captured model: NNX stream key/count updates and Linen's fixed bridge-key
behavior remain distinct. It proposes a selected Threefry representation; no
native or target acceptance is inferred from that static design.

Two closed standalone selection probes retain the original Where failures and
strict criteria. The second records 576 exact observations: 504 small saved
outputs and 72 realistic A/B/A calls bound to four distinct saved actual arrays.
There are 168 sessions, no warmups and 72 persisted timing samples. INT32
GatherElements has the better measured speed/memory tradeoff than Compress,
while still requiring a substantial two-arm stack. These finite CPU observations
do not establish production lowering or general performance. Only the two-path
SOURCE FP16/32/64 BOOL selection implementation is released in an isolated
checkout, with runtime validation pending and no accepted or integrated fix.
Admission, LEGACY, raw callbacks and failed historical artifacts remain unchanged. The first probe's missing individual
benchmark timings and the second launcher's unfiled early Python3.10 error stay
explicit evidence limits, not reconstructed receipts.

## Instance lifetime and retained failures

The first combined MAIN full run covered all 8,766 expected identities: 8,765
passed and the existing legacy DINOv3 B14 test failed with contracting dimensions
192 versus 3072. No identity was missing, unexpected or skipped. Its failure,
source freeze and later unchanged DINO-alone/module passes remain preserved.
A controlled normal-wrapper reproduction established an instance-address reuse
(ABA) cache defect. It did not observe actual address reuse in the original
full-run event, so the repair is not proof of that event's cause.

The separately reviewed two-file repair gives each live object a stable,
monotonically allocated key held through weak references. Reuse requires the
identical live object; stale cleanup cannot evict a successor. It avoids model
hashing/equality/mutation, strong lifetime retention and per-call cache resets.
Six normal-wrapper/lifetime regressions are added. All eleven original test
anchors and assertion ASTs remain, with four private address-key binds updated
to use registered keys. The owned module's seventeen cases pass on both stacks.

Owner broad execution has 828 passes on the primary; the same secondary case
set has 823 passes, four explicit compatibility skips and one strict FP16
`decoder-float16` benchmark failure. Three skips concern unavailable
ConvTranspose/BatchNorm runtime fusion; one concerns older JAX's absent
`one_minus_square`. Unchanged DINO's eight cases pass separately per stack.
These overlapping selections must not be added into one total. The controlled
old-fails run is primary-only; post-fix passes are recorded on both stacks.

The secondary FP16 failure is also present on pristine accepted HEAD `1c254d69`.
The baseline/candidate comparison retains byte-identical native reference, two
feeds, submitted model, actual target output and errors under the same settings.
Two of 96 values violate unchanged `rtol=atol=0.002`, with maximum absolute
error `0.00341796875`. Both exact-node processes exit 1. This proves that specific
failure predates the named/preparation/cache packets; it remains **FAILED** and
unresolved, not waived, skipped or declared an ONNX limit. The earlier observer
serialization failure remains a separate epoch.

## Actual MAIN integration boundary

The nine named files, separate two-file portable PoC, two-file lifetime repair
and inventory form fourteen checkpoint paths. The frozen complete union contains
871 Python files, including root `logging_config.py`; all 27 preexisting
precision Markdown files remain unchanged. This complete map is separate from
the named owner's narrower 868-file watcher.

MAIN's pre-repair focused epoch passed 2,174 cases/6,522 phases per stack.
The current post-repair epoch passes **2,191 cases/6,573 phases per stack**,
without failures, errors or skips, across the same 22 selected modules on both
stacks. Eleven modules have explicit actual MAIN import evidence; collection
records bind all 22 selected paths. Five effective and five hidden settings,
policy, 25 callable identities and the full 871/14/27 source/checkpoint/document
maps are preserved. Independent review accepts this completed integration scope;
the subsequent fresh full suite and actual code hooks are recorded below.

The inventory records **220 manual files, 1,363 functions and 611 migration
additions**, retaining all 1,357 immediate predecessor functions and all 752
original anchors, with zero original removals. Nonmanual fields, 1,802 raw rows
and 3,604 cells remain unchanged. Both MAIN inventory CLIs pass, retaining the
older stack's conditional absent `one_minus_square` entry. Inventory SHA256 is
`30c64da33c46c727ae9b6e3b33d869b3ad145404f786d9ec889816e3a0a9774b`.

Closed MAIN checks pass Black on all thirteen changed Python files, global Ruff
and the actual configured `./scripts/check_typing.sh`. Recorded versions are
Black `0.1.dev1+g8a737e727` and Ruff `0.16.10`; these are checks, not actual commit
hooks. The subsequent complete primary-stack run exits 0 with **8,772 passes**,
no errors, failures or skips, and 13,773 warnings. Pytest reports 570.05 seconds
(the XML records 569.964 seconds; wrapper time is 571.5396 seconds). Its actual
identity set exactly matches every preceding 8,766 case plus six lifetime
regressions. All 871 Python, fourteen checkpoint and 27 existing Markdown files
remain unchanged.

Code commit `8f823ac3b253aaf62415a6170ea875ec1f022f55` passes all seven actual
repository hooks without bypass, with a clean worktree and the same frozen
hashes. The first failed full epoch and baseline-identical secondary FP16
failure remain visible; a successful primary full run does not turn that
secondary failure into a pass. This checkpoint completes these bounded changes,
not the all-model migration or public precision transition.

## Binder preparation history

Independent v1 review required two shared-helper repairs: scenario-dependent
named values were revalidated under the wrong restored context, and a generic
payload check displaced an established symbolic-shape diagnostic. V2 fixes only
those helper issues and adds six configuration regressions; production semantics,
original negatives and criteria remain unchanged.

V1's secondary broad run retains 1,705 passes and one failure with its state
receipt. The primary required SIGTERM and exited 143 without a final state
receipt; restoration is not retroactively claimed. Interrupted model runs and
zero-export configuration reproductions remain historical records. Final v2
receipts verify five effective settings, five hidden globals, policy and ten
watched callable identities, plus correct package import/cwd.

The owner watcher covers four directories: base 866 Python files plus two new
binder tests gives 868. It excludes root `logging_config.py`; the complete
binder-only set is 869. This scope must not be confused with the combined MAIN
source union after adding the separate two-file preparation PoC.

Owner Black/Ruff, configured mypy on 542 sources, and the RNG registry check
pass. The configured Black hook is pinned to revision 25.1.0/source `8a737e727`;
its actual version string is `0.1.dev1+g8a737e727 (compiled: no)`. Owner checks
are distinct from actual commit hooks. Strict historical control/hash packets
are not repinned to current sources. The preceding
[Typed checkpoint](typed_values_and_masks_findings.md) and every historical
numerical failure keep their original evidence and scope.

## Frozen evidence

These local audit records are separate from the portable tracked implementation.
They identify the closed source and observations; they are not required to import
or use the binder. The [work plan](../precision_implementation_work_plan.md)
retains the full B–F completion contract and one final public user experience.

| Artifact | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-named-parameters-frozen-source-v2/manifest.json` | `cedffe9c2517d0306797867f236dfd463498170edb3fdb39850267ef51aff9f6` |
| `/tmp/jax2onnx-named-final-candidate-v2.json` | `f7263c3954c4ddffe4161ab8524f3ed5b254487d1a1c42c2a26076b8ea83afd2` |
| `/tmp/jax2onnx-named-code-astra-review-v1.md` | `34e3c4ee268328ea687c32fd340ed239be09b757b92593d0ed591bda9c453d72` |
| `/tmp/jax2onnx-named-code-astra-review-v2.md` | `387c3ef216426324d0c207a9574e5a14668cb8a872d01fff2a8b1bf7d3290e63` |
| `/tmp/jax2onnx-named-final-v2-core-evidence-closed.json` | `14ed4aa9ea8cfbaafa3e8d1ce2abb3225a7d65aee115565cfa4d8dd1bf6a0da5` |
| `/tmp/jax2onnx-named-runtime-core-astra-review-v2.md` | `ede8d7a74be5c340d4badeb56ac93ea340053736347ed35c3070c8451f5330f7` |
| `/tmp/jax2onnx-named-runtime-core-astra-verification-v2.json` | `560b274263f1c7d298538bb87ef44c1d979b789870f9134918796786f9f945a0` |
| `/tmp/jax2onnx-named-final-v2-all22-evidence-closed.json` | `d7e190bf67420e2641ecedd44372947fb5d9e20c897cb554cd8d4e7a527e2bbe` |
| `/tmp/jax2onnx-next-mlp-dependencies-plan.md` | `ca0d472a93aca60a0e1b75a0592ec2d1b07bfd6e52b051ab93c20b069edb911d` |
| `/tmp/jax2onnx-next-mlp-dependencies-bindings.json` | `158e4dfc599f759dd4fb5b420bcb3dbe96f49b3cdfd4faaad4376823fbeb632d` |
| `/tmp/jax2onnx-next-mlp-dependencies-astra-review.md` | `c15ac995ec635b61819981901eb368818e96bde1acbaa8f1f517ed97b69739ff` |
| `/tmp/jax2onnx-named-preparation-main-integration-freeze-v1.json` | `16961900f439c9e0b0ff3d0d5be0156e7b39f36f661c6a69c356869518d945a8` |
| `/tmp/jax2onnx-named-preparation-main-inventory-dual-v1.json` | `e0cb3c1dca92aeefb39b25615dfd408f9acc7c9ad8c04a6ed41cac5aaf9c4f0e` |
| `/tmp/jax2onnx-named-preparation-main-style-v1.json` | `af7221eb79c53961c3875a4dc8df9fda84328ae4b711d203e81d6c9e9b6473ce` |
| `/tmp/jax2onnx-named-parameters-final-handoff-v2.json` | `f1679c7faf4fff638f34d2850f6df01961ab74e9853b506e794c463382395c02` |
| `/tmp/jax2onnx-named-preparation-main-focused-dual-v1.json` | `f02d87f6fa6265f116d397539a70a9fea92d7e583454b93f38002bbda1f3618f` |
| `/tmp/jax2onnx-named-all22-progress-astra-review-v2.md` | `07a52b57be27da8847facd375268afa0a6c40911591849dab00e4ec3f60531e4` |
| `/tmp/jax2onnx-named-preparation-main-full-result-v1.json` | `45b36150d64f83c7dcfacecf114c9276bd17b572e45ae90663be5ee10f34b67e` |
| `/tmp/jax2onnx-named-preparation-main-full-observed-identities-v1.json` | `9c10501e65692a9a9e748d511e06242346812a76c05015a31ab9701ef3801ca2` |
| `/tmp/jax2onnx-instance-lifetime-repair-closed-v1.json` | `a40afa24fb3aad195e2b76210576c68697a6c622a97545d5548ed92a6337a278` |
| `/tmp/jax2onnx-instance-lifetime-code-astra-review-v1.md` | `df4bbc90184f76909fad5abf2153b1a1ea95151d1d5b59c11338e6172e94001b` |
| `/tmp/jax2onnx-instance-lifetime-repair-v1/benchmark-baseline-candidate-comparison.json` | `57fbf23eff516e1dbf63fbbda92c73fb317a23beaaa2228b9b213d3976247a2e` |
| `/tmp/jax2onnx-instance-main-integration-astra-review-v2.md` | `95f3243e0491b718effde4f017d287ed93656af125c8e5955769ec7c4eff6f3d` |
| `/tmp/jax2onnx-named-preparation-instance-main-integration-freeze-v2.json` | `4e1eccc6c53d8c85021907d04d5a1d7944d5411b1d37ad4605a7ee43c533c68d` |
| `/tmp/jax2onnx-named-preparation-instance-main-focused-dual-v2.json` | `40e7e9da58ea680c6c930126c8176479095bc7ae5cdc8b7642e7f8d34f8ea11a` |
| `/tmp/jax2onnx-named-preparation-instance-main-checks-v2.json` | `59720b10e96f1a11172866fee9f95c5b83c24b331aa7f2c839783c8386f89782` |
| `/tmp/jax2onnx-next-mlp-state-prng-static-supplement.md` | `16a9fa255f186a1ffc4cfe0ad973807321fdeb1c4c6058d432afa3be56df2799` |
| `/tmp/jax2onnx-where-copy-probe-astra-review-v1.md` | `e010670a28b929169c30c9d931d905b06dfa401ad7001003c718f2fe162a10c3` |
| `/tmp/jax2onnx-where-cheaper-probe-astra-review-v1.md` | `2bb8435040ae9fe47d20404402a8f70ba9765e56d8d2b660e368d4684cd94a52` |
| `/tmp/jax2onnx-exact-select-release-v1.json` | `f17a9586702f61370dc004470841bb3389fe4f439abcd6609a5d59e062ede25a` |
| `/tmp/jax2onnx-named-preparation-instance-main-full-result-v2.json` | `fcdae33751b0de7b97da860800a788ed61ec5a039b81dee2bb582041a3d61954` |
| `/tmp/jax2onnx-named-preparation-instance-main-code-checkpoint-v2.json` | `3e920127e90f6c011f0d72c1769f4667fda0c517ed4d7065ddcc7dd9086ea6c1` |
| `/tmp/jax2onnx-mlp-closure-release-v1.json` | `d5666a07bf42c4ba8bc12083c299650a9e7bc9a0a6bf14aec31310ea095e325e` |
