# Native Exp and Leading-B Mul Findings

Status: finite private semantics and actual-main integration independently
accepted, 2026-10-18. The four-path packet preserves native `jnp.exp` calls and
adds bounded floating Exp/Mul composition to the existing audited SOURCE path.
Both actual-main selections pass 1,489 cases. The separate independent review
passes 153 cases per stack. The code is committed at
`7c68300b29319844fb691c64a841092a3c2e3471`; the full primary-stack run passes
6,933 tests and all seven actual commit hooks pass.
Public LEGACY behavior, the `source-static-v1` identifier and report schema 1
remain unchanged.

The prerequisite [Leading-B matrix/bias checkpoint](leading_batch_matrix_bias_findings.md)
is committed at `718f448e`, with its own accepted forty-run FP32 observation on
each stack. That packet’s earlier preparation failures, complete saved outcomes
and wrapper post-return error remain historical records. This checkpoint adds
neither a new five-row nor a new twelve-row numerical execution release.

## Source and representation contract

The only production change to the native NumPy Exp plugin is the existing
binder’s private `source_native=True` opt-in. SOURCE forwards the original
arguments and keyword errors to native JAX Exp. LEGACY/custom lowering,
abstract evaluation, registration and differentiation retain their prior AST.
The two changed binary-dispatch guards now count the sixth reviewed native
caller; the Min/Max exclusions remain unchanged.

| Operation | Bounded audited contract |
| --- | --- |
| Exp | One floating input and one equal-dtype result, exact shape and weak type, absent/None accuracy, unspecified placement and closed parameters. FP16/BF16/FP32/FP64 structure is covered. |
| Static Mul | Existing native floating elementwise composition, retaining explicit source broadcasting, casts and the traced output properties. |
| Leading-B Mul | Equal-rank two/three operands with equal positive static tails; the same actual native B or one explicitly static leading 1. Result weakness is the conjunction of operand weakness. |

The actual native symbol scope governs B equality. Two declared B operands
cannot use unrelated scopes or different concrete batches because a target
could broadcast them. A declared static one remains a static one. Existing
source boundary validation rejects mismatched feed/reference shapes before
creating a target session and checks the bound output after a call.

Exp and Mul preserve native parameter closure, purity, arity, result types and
placement restrictions. Non-None Mul placement is rejected. No new Where,
general symbolic shape, JIT/control-flow or diagnostic-validator admission is
included. Add/MatMul/bias-reshape contracts retain their accepted boundaries.
Canonical source operands are not mutated and no implicit arithmetic Cast is
introduced. Source-authored casts and `broadcast_in_dim` remain real graph
operations, rather than being erased to satisfy a simplified expectation.

Named/nested composition covers actual formal mappings, aliases, captures and
cache order. One serialized artifact serves B=2 and B=5. Grad/vmap artifacts,
local x64 conflicts and restoration after rejected local scopes are included.
These are finite representations and examples, not blanket operator or model
acceptance.

## Fixed numerical and structural evidence

Independent unpatched native eager/JIT references precede export. Test criteria
are fixed before target output: bitwise controls for exactly representable
Exp special values and sign-sensitive Mul; exact NaN-class checks where payload
identity is not promised; and finite FP32 Exp comparisons at rtol=atol=1e-5 or
FP64 at rtol=atol=1e-7. Captured NaN/zero payload preservation and runtime
NaN-class equality are separate claims. No tolerance is widened after a failure.

The independent review includes nested rank-three FP64 Exp/Mul/Add and actual
MatMul/bias/Add followed by Exp/Mul, named functions and returned aliases,
opsets 23/26, and CPU optimization disabled/all. It preserves the same artifact
for both batches and checks immutable input/artifact bytes. Selected finite
CPU samples pass. BF16 evidence is limited to source/schema/special-value
structure; it does not establish a working CPU kernel. There is no universal
Exp accuracy, arbitrary low-precision tolerance, GPU or whole-model guarantee.

Negative cases reject malformed dtype/shape/weakness, forbidden symbolic
families, nondefault accuracy/placement, unrelated native scopes and bad sample
bindings. Strict target mismatch remains FAILED and the acceptance gate raises.
Existing Dropout negative controls and diagnostic Exp exclusion are retained.

The first owner-focused run records 194 passes and two assertion failures.
Native static rank promotion contains an explicit `broadcast_in_dim(shape=(1,3))`;
the artifact must therefore retain its Expand. Only the test expectation was
corrected to require this native correspondence and no Cast. The original
failure record and fixed criteria remain; the corrected selection passes 196.
This is not a production rewrite or a waived numerical failure.

## Independent and actual-main validation

| Selection | Python 3.12 / JAX 0.11.2 | Python 3.11 / JAX 0.10.2 |
| --- | --- | --- |
| Owner complete focused selection | 1,489 passed | 1,489 passed |
| Independent semantic review | 153 passed | 153 passed |
| Actual MAIN integration | 1,489 passed, 145 warnings, 305.30 s | 1,489 passed, 136 warnings, 230.10 s |

The 153 independent cases comprise 117 new permanent cases, 25 independent
review guards, two updated binary guards and nine unchanged Dropout negative
controls. The 1,489-case selections retain all previous 990 integration cases
and fourteen frozen Leading-B review guards, plus the new Pointwise/native
dispatch anchors; generated Exp/Mul classes account for seventeen cases.
Counts overlap across selections and must not be added as distinct coverage.

Actual MAIN execution is bound to predecessor
`d85f61789f7704b57f04a72fd6671d28935a697c` and the exact accepted four paths.
Both collections have digest
`951983902491599a4f712015ebadb724661dd59b712ef21573d31246c4d41bd6`.
All thirty imported test modules match MAIN or the four declared immutable
external files. Each execution has 1,489 distinct calls and 4,467 passing
setup/call/teardown reports; XML has no failures, errors or skips.
All 860 Python source hashes agree before and after. The explicit CPU backend,
one-thread settings, pytest `-n0`, five effective/hidden JAX settings, native
JNP/NNX callables and LEGACY policy are verified. Generated output working
directories alone move to /tmp; source imports remain bound to MAIN.

The complete primary-stack run passes **6,933 tests** with 13,130 warnings
in 403.51 pytest seconds (404.855 seconds for the wrapper), with no failures,
errors or skips. All five frozen code/inventory paths and all 860 Python
source hashes remain unchanged.

Whole-package mypy passes 542 files. Configured Black, Ruff, IR/reflection,
annotation, path and RNG checks pass in the owner packet. Independent review
accepts the four-file semantics and separate actual-main integration.
Commit `7c68300b29319844fb691c64a841092a3c2e3471` passes all seven configured
hooks without edits, skips or bypass. The five code/inventory paths and all
860 Python hashes remain unchanged after hooks; the checkout is clean at
the recorded code checkpoint.

The refreshed inventory contains 210 files, 1,217 functions and 465 migration
additions. All 752 original anchors remain, with zero original removals and
unchanged 1,802 raw contracts/3,604 scenario cells. Only counts, manual tests and
manual provenance change. Twenty new functions account for the growth; the
migration-only guard is renamed from five to six reviewed callers. Both actual
inventory CLI comparisons pass, retaining the secondary stack’s conditional
`one_minus_square` baseline entry. These comparisons establish inventory
identity, not SOURCE acceptance for every row.

## Preserved histories and remaining work

The historical numerical loader and its strict provenance are not repinned.
Negative preflight on each stack rejects the changed source/tooling provenance
with exit2 before preparation or ORT; it creates no numerical output and leaves
the old approval/results unchanged. The old raw None criteria, fixed twelve-row
approval and five-row Leading-B observation remain intact. Existing subnormal,
signed-zero, BF16 and other strict target failures keep their original status
and raising gates. New fixed Pointwise tests do not rewrite those histories.

Where/JIT/select is the next separately authored follow-up, not yet accepted
or integrated at this checkpoint. Native JAX uses version-dependent typed
inlining-policy forms; the measured enum/Boolean compatibility boundary needs
an exact compatibility seam and independent review. The recorded x64 conflict
and the original gather source still require their own evidence. An unfinished
implementation is not a demonstrated technical impossibility.

After the separate Where/JIT review, group remaining work by coherent operator
families and their shared representation rules, while retaining the full
source-preservation and validation gates. Individual opt-ins are intermediate
steps, not the completion criterion.

The broader 310-row symbolic inventory, seven NCHW rows, all 22 parameterized
models and their actual runtime controls, normalization, metadata-free
registrations and optional MaxText/MaxDiffusion discovery remain part of phases
D/E/F. The goal is one normal source-preserving interface across existing fixed
source implementations, including FP64. Public LEGACY and both explicit
`enable_double_precision` Boolean values keep their present behavior until the
reviewed final phase-F transition; no public selector or permanent per-operation
LEGACY fallback is introduced here.

## Frozen evidence

The records below retain their own source bindings and outcomes. The owner
summary predates the later independent reviews and MAIN integration; its old
pending status is not the current status of this checkpoint.

| Evidence | Local artifact and SHA256 |
| --- | --- |
| Accepted four-path packet | `/tmp/jax2onnx-pointwise-mul-exp-four-files.patch`, `f7ee7888071f0b87bae4f80a62236f906af194c6d780cbc60841b2e75c799bb5` |
| Owner validation and preserved first-run ledger | `/tmp/jax2onnx-pointwise-mul-exp-final-summary.json`, `b5f4af7fb667b4ee50a25dd993706d35a0ca43314a7519784d28fc9a2000150a` |
| Independent semantic acceptance | `/tmp/jax2onnx-pointwise-mul-exp-astra-review.md`, `dc8516482ee70217b2953ae1bfac903ddfda1b8dc9cc9ec5c49d8af01e08930b` |
| Independent Python 3.12 selection | `/tmp/jax2onnx-pointwise-astra-review-evidence/py312.json`, `822dd33bc64366a9858a45d54360a5bb3e96058f4d5a330e05ac17a19a954d52` |
| Independent Python 3.11 selection | `/tmp/jax2onnx-pointwise-astra-review-evidence/py311.json`, `ca1a874557bebebef23406778000be036971d4b5d809d0a113f1b031330f8eee` |
| Actual-main integration and inventory acceptance | `/tmp/jax2onnx-pointwise-main-integration-astra-review.md`, `07168672b0879428c5bf0f4c0e3933c473137c72a1f8affaee434977bc86f661` |
| Both actual-main verification records | `/tmp/jax2onnx-pointwise-main-dual-verification.json`, `31059306247461366b0b7a9568355d8d783bfce0fa0bdb9b67ee16f4fc8622dd` |
| Actual-main Python 3.12 execution | `/tmp/jax2onnx-pointwise-main-validation/py312-execution.json`, `3bc4f4addf8df86a7cf89b4244a51ea5bd0fd8d44f1196bbd66a114fc1900528` |
| Actual-main Python 3.11 execution | `/tmp/jax2onnx-pointwise-main-validation/py311-execution.json`, `6537ab621b7c653800b301af95a1b0dc434fde5ae41a51c45fae49f901f416eb` |
| Exact refreshed inventory | `/tmp/jax2onnx-pointwise-inventory-candidate.json`, `b6b9c3b9544c0daa4076957e40e0be22bb581773f6e465466a1a8d202913463f` |
| Independent inventory and predecessor audit | `/tmp/jax2onnx-pointwise-inventory-astra-audit.json`, `e6cb91e82dc5c234870e343d521ee560d7bae7975d0f9c06ec15fcfcac322fa5` |
| Complete primary-stack run | `/tmp/jax2onnx-pointwise-main-full-result.json`, `905a83578e5260f2e18ccc73b140562e1129b515b7a4f6710937a203807c172b` |
| Five-path code/inventory freeze | `/tmp/jax2onnx-pointwise-main-full-freeze.json`, `daada9d43385b02e2c05087db4d643c26352055441f1ac02387505c248f7e711` |
| Preserved Leading-B numerical acceptance | `/tmp/jax2onnx-leading-b-main-runtime-astra-review.md`, `1ca74875bf9e6b7c5974bc488f9142d974682c154dac1b798bf8eaabb66ab53a` |
| Preserved Leading-B outcomes and wrapper error | `/tmp/jax2onnx-leading-b-main-runtime-reconciliation.json`, `a296fb762457d2c11bfd078ddb7f1f7d906d657896fe4dca3e64f2d81f573aed` |
| Committed code, actual hooks and frozen hashes | `/tmp/jax2onnx-pointwise-main-code-checkpoint.json`, `aa48a7517f5cff6a19bd473f8bb336fd8e5001a3414ca3fb9c32e39c7da3f3cb` |
| Actual seven-hook commit log | `/tmp/jax2onnx-pointwise-main-code-commit.log`, `72f0c8aafd1e88fc6e5c26fc7d00d66e0a0faf9c393987a25e14a910f08648b3` |
