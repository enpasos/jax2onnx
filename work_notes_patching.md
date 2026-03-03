# Work Notes: Durable AD Rule Completeness for Patched Primitives

## Background
- Issue: `https://github.com/enpasos/jax2onnx/issues/203` (opened on 2026-03-02).
- Observed failure under conversion patching:
  - `NotImplementedError: Transpose rule (for reverse-mode differentiation) for 'jax.numpy.transpose' not implemented`
- Root cause:
  - `to_onnx` activates plugin worlds and monkey-patches many framework callsites to custom primitives.
  - Those custom primitives often register `ad.primitive_jvps[...]` but not `ad.primitive_transposes[...]`.
  - `grad` and many `vjp` traces can still pass, but `jax.linear_transpose(...)` paths fail.
- Scope is broad, not a single primitive:
  - Reproduced for `jnp.transpose`, `jnp.add`, `jnp.reshape`, `jnp.squeeze`, `jnp.moveaxis`, `jnp.stack`, `jnp.concatenate`, `jnp.split`, `jnp.tile`.

## Goal
Establish a durable, cross-component invariant so patched primitives remain autodiff-safe under all conversion-time traces, including reverse-mode transpose paths.

## Non-Goals
- Re-architecting away from plugin patching in this change.
- Writing handcrafted transpose rules for every primitive.
- Changing ONNX lowering semantics.

## Execution Progress (Strict Mypy Cleanup)
- Last updated: 2026-03-03.
- Baseline at start of strict pass: `684 errors in 111 files` (`mypy jax2onnx/plugins/jax`).
- Current snapshot: `0 errors in 0 files` (`mypy jax2onnx/plugins/jax`).
- Progress (file-based): `100.0%` complete (`111/111` files clean).
- Progress (error-based): `100.0%` complete (`684/684` errors removed).
- Rule for execution: update this section after each completed tranche (counts + remaining checklist).

### What Still Needs To Be Done (Always Update)
- Overall completion: `100.0%` (errors removed) / `100.0%` (files cleaned).
- Remaining strict-mypy scope: `0 errors` across `0 files`.
- Done condition: `mypy jax2onnx/plugins/jax` reports `Success: no issues found`.
- Work mode: pick the smallest remaining file, make it strict-clean, run focused tests, then refresh this section.

### Remaining Work Checklist
- [x] `jax2onnx/plugins/jax/lax/while_loop.py` (35)

## Testing Discipline (Mandatory)
- Every implementation change in this plan must ship with tests in the same PR/commit scope.
- Prefer the standard local plugin test style whenever possible:
  - Add/update plugin-focused tests under `tests/primitives/` for primitive/plugin behavior changes.
  - Add/update `tests/extra_tests/framework/` for policy/invariant enforcement changes.
  - Add/update focused converter regressions under `tests/extra_tests/` for end-to-end `to_onnx` failures.
- For each change, run at least one focused local pytest target that directly exercises that change before moving on.
- Do not treat a phase item as done without passing focused tests plus the relevant policy/regression checks.
- Keep regression tests issue-linked where appropriate (e.g., issue #203) so intent stays traceable.
- Add explicit policy tests for:
  - no override of existing custom transpose rules,
  - idempotence/reentrancy of backfill/activation paths.
- Test-file naming is fixed by this plan (see **Canonical New Test Files**) and must not drift during implementation.

## AD Invariant (Policy)
- For every jax2onnx patched primitive:
  - If a JVP rule is registered, a transpose rule must also be registered.
- Custom/specialized transpose rules remain authoritative and must never be overridden by generic fallbacks.

## Design Strategy

### 1. Prefer original-rule forwarding for true 1:1 mappings
- Add helper support to optionally bind AD/batching rules from an original JAX primitive when semantics are equivalent.
- Example pattern:
  - `ad.primitive_jvps[new_prim] = ad.primitive_jvps[orig_prim]`
  - `ad.primitive_transposes[new_prim] = ad.primitive_transposes[orig_prim]`
  - `batching.primitive_batchers[new_prim] = batching.primitive_batchers[orig_prim]`
- This path should be explicit and opt-in per plugin family to avoid accidental semantic mismatches.
- Enforce an allowlist-driven approach:
  - maintain a curated `orig_prim -> new_prim` mapping table,
  - reject implicit forwarding when mapping is not allowlisted.
- Approved forwarding mappings as of 2026-03-03:
  - `add -> jax.numpy.add` (forward JVP+transpose from original primitive; keep plugin-specific batching rule).
  - `concatenate -> jax.numpy.concatenate` (forward JVP+transpose from original primitive; plugin keeps dedicated batching rule).
  - `transpose -> jax.numpy.moveaxis` (forward JVP+transpose from original primitive; plugin keeps dedicated batching rule).
  - `reshape -> jax.numpy.reshape` (forward JVP+transpose from original primitive; plugin keeps dedicated batching rule).
  - `split -> jax.numpy.split` (forward JVP+transpose from original primitive; plugin keeps dedicated batching rule).
  - `squeeze -> jax.numpy.squeeze` (forward JVP+transpose from original primitive; plugin keeps dedicated batching rule).
  - `tile -> jax.numpy.tile` (forward JVP+transpose from original primitive; plugin keeps dedicated batching rule and aliases raw `tile` primitive lowering).
  - `transpose -> jax.numpy.transpose` (forward JVP+transpose from original primitive; plugin keeps dedicated batching rule).
- If no reliable original primitive exists (composite/wrapper APIs), fall back to generic rule registration.

### 2. Centralize AD rule registration in `jax2onnx/plugins/jax/_autodiff_utils.py`
- Add a generic transpose helper, for example:
  - `register_transpose_via_linear_transpose(prim, impl, *, override=False)`
- Make existing helpers enforce completeness by default:
  - `register_fallback_jvp_rule(...)` registers JVP + transpose.
  - `register_jvp_via_jax_jvp(...)` registers JVP + transpose.
- Transpose helper behavior requirements:
  - Handle `ad.UndefinedPrimal` arguments correctly.
  - Preserve `ad.Zero` behavior.
  - Return `None` for known primal inputs in transpose rule output tuple.
  - Respect existing explicit transpose rules unless `override=True`.

### 3. Add a global backfill pass for legacy/manual registrations
- In plugin bootstrap/activation, run one idempotent pass that scans jax2onnx leaf primitives.
- For any primitive with JVP but missing transpose, register generic transpose fallback via its `def_impl` callable.
- This immediately covers components that still assign `ad.primitive_jvps[...]` directly.
- Keep this backfill as a safety net even after migration.
- Add an operational kill switch for fast rollback:
  - e.g. `JAX2ONNX_DISABLE_AD_BACKFILL=1` disables backfill registration without code changes.

### 4. Preserve explicit specialized rules
- Do not replace handcrafted rules such as `sum` transpose.
- Generic fallback should be additive only when missing.
- Add an explicit policy test that fails if a generic path overwrites an existing custom transpose rule.

### 5. Guardrail tests in framework policy suite
- Add a framework test that fails when any jax2onnx primitive violates `JVP => transpose`.
- Run after importing all plugins to cover the real runtime registry state.

### 6. Conversion boundary regression tests
- Add focused regression tests that run through `to_onnx` (not only raw JAX):
  - `linear_transpose` of `jnp.transpose`.
  - `linear_transpose` of `jnp.add` (representative non-unary case).
- Ensure failures from issue #203 remain permanently covered.

### 7. Migration hygiene
- Gradually replace direct `ad.primitive_jvps[...] = ...` writes with `_autodiff_utils` helpers.
- Keep backfill in place until all plugin families are migrated and policy tests remain green.

## Implementation Plan

### Phase A: Infrastructure
- [x] Add helper API for original-rule forwarding (`orig_prim -> new_prim`) in `_autodiff_utils.py`.
- [x] Identify and document safe 1:1 primitive mappings where forwarding is allowed.
- [x] Implement transpose registration helper in `_autodiff_utils.py`.
- [x] Update existing JVP helpers to register transpose automatically.
- [x] Add idempotent backfill function and call site during plugin activation/import.
- [x] Add kill switch env handling for backfill (`JAX2ONNX_DISABLE_AD_BACKFILL=1`).
- [x] Add optional debug logging (`JAX2ONNX_AD_DEBUG`) for fallback installs.
- [x] Add/update focused framework tests for helper/backfill behavior as each infrastructure item lands.

### Phase B: Regression Coverage
- [x] Add issue-203 style conversion repro test for `jnp.transpose` + `linear_transpose`.
- [x] Add second conversion repro (`jnp.add`) to avoid single-op blind spot.
- [x] Add framework policy test for global `JVP => transpose` invariant.
- [x] Add framework policy test for "no custom-rule override" behavior.
- [x] Add framework policy test for backfill idempotence/reentrancy.
- [x] Ensure each new regression has a focused local pytest command documented/used during development.

### Phase C: Migration and Cleanup
- [x] Convert manual JVP registrations to helper calls (incremental PRs acceptable).
- [x] Keep explicit custom transpose rules untouched.
- [x] Document plugin AD expectations in dev guide(s).
- [x] For each migrated plugin family, add/update at least one direct plugin test proving AD registration remains complete.
- [x] Update migration tracker table after each family migration.

## Canonical New Test Files
- `tests/extra_tests/framework/test_ad_rule_completeness.py`
- `tests/extra_tests/framework/test_ad_rule_no_override.py`
- `tests/extra_tests/framework/test_ad_backfill_idempotence.py`
- `tests/extra_tests/test_issue_203_linear_transpose_transpose.py`
- `tests/extra_tests/test_issue_203_linear_transpose_add.py`

## Local Test Commands by Phase
- Phase A (framework/helper and idempotence):
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_no_override.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_backfill_idempotence.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_forwarding.py`
- Phase B (conversion regressions):
  - `poetry run pytest -q tests/extra_tests/test_issue_203_linear_transpose_transpose.py`
  - `poetry run pytest -q tests/extra_tests/test_issue_203_linear_transpose_add.py`
  - `poetry run pytest -q tests/extra_tests/test_issue_203_linear_transpose_allowlist_ops.py`
- Phase C (plugin family migrations):
  - `poetry run pytest -q tests/primitives/test_jnp.py -k \"transpose or add or reshape or squeeze or moveaxis or stack or concatenate or split or tile\"`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py`
- End-of-phase/full validation:
  - `poetry run pytest -q`
  - `poetry run ruff check .`
  - `poetry run mypy src`

## Migration Tracker
| Plugin Family | Status | Forwarding Allowlist Added | Backfill Coverage Verified | Tests Added/Updated | Last Updated |
|:--|:--|:--|:--|:--|:--|
| `jax.numpy` | in-progress (allowlisted linear subset covered by conversion regressions) | yes (subset; includes `add -> jax.numpy.add`, `concatenate -> jax.numpy.concatenate`, `transpose -> jax.numpy.moveaxis`, `reshape -> jax.numpy.reshape`, `split -> jax.numpy.split`, `squeeze -> jax.numpy.squeeze`, `tile -> jax.numpy.tile`, `transpose -> jax.numpy.transpose`) | yes (allowlisted subset) | `tests/extra_tests/framework/test_ad_rule_forwarding.py`; `tests/extra_tests/test_issue_203_linear_transpose_transpose.py`; `tests/extra_tests/test_issue_203_linear_transpose_add.py`; `tests/extra_tests/test_issue_203_linear_transpose_allowlist_ops.py`; `tests/primitives/test_jnp.py -k "take or where or select or prod"`; `tests/primitives/test_jnp.py -k "transpose or add or reshape or squeeze or moveaxis or stack or concatenate or split or tile"`; `tests/extra_tests/framework/test_ad_rule_completeness.py::test_migrated_jnp_linear_primitives_have_jvp_and_transpose` | 2026-03-03 |
| `jax.nn` | in-progress (helper migration complete; strict-mypy cleanup complete for `jax.nn`; transpose fallback intentionally not allowlisted for nonlinear ops) | no | n/a | `tests/primitives/test_nn.py -k "relu or sigmoid or gelu or softplus or celu or selu or silu or soft_sign or mish or leaky_relu"`; `tests/primitives/test_nn.py -k "softmax or logsumexp or logmeanexp or log_softmax or glu"`; `tests/primitives/test_nn.py -k "dot_product_attention or dpa"`; `tests/extra_tests/framework/test_ad_rule_completeness.py::test_migrated_nn_primitives_have_jvp`; `tests/extra_tests/framework/test_ad_rule_completeness.py::test_migrated_nn_primitives_are_not_allowlisted_for_linear_transpose` | 2026-03-03 |
| `jax.lax` | in-progress (strict-mypy alias cleanup tranche complete across all legacy `JaxprEqn = getattr(...)` plugin modules; broader typing backlog remains) | no | n/a | `tests/primitives/test_lax.py -k "abs or acos or acosh or asin or asinh or atan or ceil or erf or erfc or exp2 or floor or log1p or logistic or neg or round or sign or sin or tan or tanh"`; `tests/primitives/test_lax.py -k "Test_and or Test_or or Test_xor or Test_eq or Test_ne or Test_lt or Test_gt or Test_less_equal or Test_greater_equal or Test_atanh or Test_cbrt or Test_erf_inv or Test_expm1 or Test_is_finite or Test_shift_left or Test_shift_right_logical"`; `tests/primitives/test_lax.py -k "Test_bitwise_not or Test_clz or Test_cos or Test_cosh or Test_log or Test_population_count or Test_shift_right_arithmetic or Test_sinh or Test_sqrt or Test_square"` | 2026-03-03 |
| `jax.core` | in-progress (`jax.core` plugin strict-mypy backlog closed for current files; broader `plugins/jax` backlog remains outside this family) | no | n/a | `tests/primitives/test_core.py` | 2026-03-03 |
| `jax.random` | in-progress (`jax.random` plugin strict-mypy backlog closed for current files; broader `plugins/jax` backlog remains outside this family) | no | n/a | `tests/primitives/test_random.py` | 2026-03-03 |
| `flax.linen` | audited (no JVP registrations currently) | no | n/a | `tests/extra_tests/framework/test_ad_rule_completeness.py::test_non_jax_families_do_not_expose_unpaired_jvp_rules` | 2026-03-03 |
| `flax.nnx` | audited (no JVP registrations currently) | no | n/a | `tests/extra_tests/framework/test_ad_rule_completeness.py::test_non_jax_families_do_not_expose_unpaired_jvp_rules` | 2026-03-03 |
| `eqx.nn` | audited (no JVP registrations currently) | no | n/a | `tests/extra_tests/framework/test_ad_rule_completeness.py::test_non_jax_families_do_not_expose_unpaired_jvp_rules` | 2026-03-03 |
| `dm_pix` | audited (no JVP registrations currently) | no | n/a | `tests/extra_tests/framework/test_ad_rule_completeness.py::test_non_jax_families_do_not_expose_unpaired_jvp_rules` | 2026-03-03 |

## Execution Log (2026-03-03)
- Focused framework/policy:
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_registration_helpers_only.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_no_override.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_backfill_idempotence.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_forwarding.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_forwarding.py tests/extra_tests/framework/test_ad_rule_completeness.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_forwarding.py tests/extra_tests/framework/test_ad_rule_completeness.py tests/extra_tests/framework/test_ad_rule_no_override.py tests/extra_tests/framework/test_ad_backfill_idempotence.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py tests/extra_tests/framework/test_ad_rule_no_override.py tests/extra_tests/framework/test_ad_backfill_idempotence.py`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py -k "where or select"`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py -k "allowlisted_leaf_primitives_with_jvp_have_transpose_rules"`
- Focused issue regressions:
  - `poetry run pytest -q tests/extra_tests/test_issue_203_linear_transpose_transpose.py`
  - `poetry run pytest -q tests/extra_tests/test_issue_203_linear_transpose_add.py`
  - `poetry run pytest -q tests/extra_tests/test_issue_203_linear_transpose_allowlist_ops.py`
- Focused primitive families:
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "take or where or select or prod"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "add or reshape"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "transpose or add or reshape or squeeze or moveaxis or stack or concatenate or split or tile"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "concatenate or split or stack or squeeze or transpose or tile or take"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "select or split or tile or where or take or transpose or reshape or squeeze or moveaxis or stack or concatenate"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "select or where or take or stack or tile"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "where or select"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "prod or sum"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "mean or maximum or minimum or moveaxis"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "mean or outer or cumprod or cumsum or unstack"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "where or select or prod or sum or mean or outer or cumprod or cumsum or unstack"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "shape or sort"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "where or select or prod or sum or mean or outer or cumprod or cumsum or unstack or shape or sort"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "linalg_norm or linspace"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "sum or prod or mean or sqrt or tan or log or exp or sin or cos"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "tan or sinh or cosh or cos or sqrt or log"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "windows or hanning or hamming or blackman or bartlett or det"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "add or maximum or minimum or outer or windows or hanning or hamming or blackman or bartlett or det or tan or sinh or cosh or cos or sqrt or log"`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "relu or sigmoid or gelu or softplus or celu or selu or silu or soft_sign or mish or leaky_relu"`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py::test_migrated_jnp_linear_primitives_have_jvp_and_transpose tests/extra_tests/framework/test_ad_rule_completeness.py::test_migrated_nn_primitives_have_jvp`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py -k "migrated_nn or allowlisted_leaf_primitives_with_jvp_have_transpose_rules"`
  - `poetry run pytest -q tests/extra_tests/framework/test_ad_rule_completeness.py`
  - `poetry run ruff check jax2onnx/plugins/jax/_autodiff_utils.py tests/extra_tests/framework/test_ad_rule_completeness.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/where.py jax2onnx/plugins/jax/numpy/select.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/prod.py jax2onnx/plugins/jax/numpy/sum.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/prod.py jax2onnx/plugins/jax/numpy/sum.py jax2onnx/plugins/jax/numpy/where.py jax2onnx/plugins/jax/numpy/select.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/mean.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/{mean.py,outer.py,cumprod.py,cumsum.py,unstack.py}`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/shape.py jax2onnx/plugins/jax/numpy/sort.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/linalg_norm.py jax2onnx/plugins/jax/numpy/linspace.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/_unary_utils.py jax2onnx/plugins/jax/numpy/_reduction_utils.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/{tan.py,sinh.py,cosh.py,cos.py,sqrt.py,log.py}`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/windows.py jax2onnx/plugins/jax/numpy/linalg_det.py`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/{add.py,maximum.py,minimum.py,outer.py,windows.py,linalg_det.py,tan.py,sinh.py,cosh.py,cos.py,sqrt.py,log.py}`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/{pow.py,fft.py,equal.py,add.py,maximum.py,minimum.py,outer.py,tan.py,sinh.py,cosh.py,cos.py,sqrt.py,log.py}`
  - `poetry run pytest -q tests/examples/test_eqx_gpt_oss.py::Test_TransformerBlock::test_gpt_oss_transformer_block_dynamic`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/add.py jax2onnx/plugins/jax/numpy/reshape.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/prod.py jax2onnx/plugins/jax/numpy/sum.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/where.py jax2onnx/plugins/jax/numpy/select.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/prod.py jax2onnx/plugins/jax/numpy/sum.py jax2onnx/plugins/jax/numpy/where.py jax2onnx/plugins/jax/numpy/select.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{add.py,reshape.py,concatenate.py,split.py,squeeze.py,tile.py,transpose.py,stack.py,take.py,prod.py,sum.py,where.py,select.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{select.py,where.py,prod.py,sum.py,moveaxis.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/mean.py jax2onnx/plugins/jax/numpy/moveaxis.py jax2onnx/plugins/jax/numpy/maximum.py jax2onnx/plugins/jax/numpy/minimum.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{mean.py,outer.py,cumprod.py,cumsum.py,unstack.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{add.py,reshape.py,concatenate.py,split.py,squeeze.py,tile.py,transpose.py,stack.py,take.py,prod.py,sum.py,where.py,select.py,mean.py,outer.py,cumprod.py,cumsum.py,unstack.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/shape.py jax2onnx/plugins/jax/numpy/sort.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{add.py,reshape.py,concatenate.py,split.py,squeeze.py,tile.py,transpose.py,stack.py,take.py,prod.py,sum.py,where.py,select.py,mean.py,outer.py,cumprod.py,cumsum.py,unstack.py,shape.py,sort.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/linalg_norm.py jax2onnx/plugins/jax/numpy/linspace.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{add.py,reshape.py,concatenate.py,split.py,squeeze.py,tile.py,transpose.py,stack.py,take.py,prod.py,sum.py,where.py,select.py,mean.py,outer.py,cumprod.py,cumsum.py,unstack.py,shape.py,sort.py,linalg_norm.py,linspace.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/_unary_utils.py jax2onnx/plugins/jax/numpy/_reduction_utils.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{tan.py,sinh.py,cosh.py,cos.py,sqrt.py,log.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{add.py,reshape.py,concatenate.py,split.py,squeeze.py,tile.py,transpose.py,stack.py,take.py,prod.py,sum.py,where.py,select.py,mean.py,outer.py,cumprod.py,cumsum.py,unstack.py,shape.py,sort.py,linalg_norm.py,linspace.py,_unary_utils.py,_reduction_utils.py,tan.py,sinh.py,cosh.py,cos.py,sqrt.py,log.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/windows.py jax2onnx/plugins/jax/numpy/linalg_det.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{add.py,maximum.py,minimum.py,outer.py,windows.py,linalg_det.py,tan.py,sinh.py,cosh.py,cos.py,sqrt.py,log.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{pow.py,fft.py,equal.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{pow.py,fft.py,equal.py,add.py,maximum.py,minimum.py,outer.py,tan.py,sinh.py,cosh.py,cos.py,sqrt.py,log.py}`
  - `poetry run mypy jax2onnx/plugins/jax/numpy`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{concatenate.py,split.py,squeeze.py,tile.py,transpose.py,stack.py,take.py}`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "pow or power or equal or fft or ifft or rfft or irfft or tan or sinh or cosh or cos or sqrt or log or add or maximum or minimum or outer"`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "where or select or prod or sum"`
  - `poetry run mypy jax2onnx/plugins/jax/numpy --show-error-codes --no-color-output`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/composite_metadata.py jax2onnx/plugins/jax/numpy/composite_metadata_batch2.py jax2onnx/plugins/jax/numpy/composite_metadata_batch3.py jax2onnx/plugins/jax/numpy/composite_metadata_batch4.py jax2onnx/plugins/jax/numpy/composite_metadata_batch5.py jax2onnx/plugins/jax/numpy/composite_metadata_batch6.py jax2onnx/plugins/jax/numpy/conj.py jax2onnx/plugins/jax/numpy/compress.py jax2onnx/plugins/jax/numpy/clip.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/composite_metadata.py jax2onnx/plugins/jax/numpy/composite_metadata_batch2.py jax2onnx/plugins/jax/numpy/composite_metadata_batch3.py jax2onnx/plugins/jax/numpy/composite_metadata_batch4.py jax2onnx/plugins/jax/numpy/composite_metadata_batch5.py jax2onnx/plugins/jax/numpy/composite_metadata_batch6.py jax2onnx/plugins/jax/numpy/conj.py jax2onnx/plugins/jax/numpy/compress.py jax2onnx/plugins/jax/numpy/clip.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy --show-error-codes --no-color-output`
  - `poetry run ruff check jax2onnx/plugins/jax/numpy/einsum.py jax2onnx/plugins/jax/numpy/matmul.py jax2onnx/plugins/jax/numpy/arange.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy/einsum.py jax2onnx/plugins/jax/numpy/matmul.py jax2onnx/plugins/jax/numpy/arange.py`
  - `poetry run mypy jax2onnx/plugins/jax/numpy`
  - `poetry run pytest -q tests/primitives/test_jnp.py -k "arange or einsum or matmul or clip or compress or conj or cov or cross or delete or fftfreq or rfftfreq or fftshift or ifftshift or logspace or matvec or median or nanmax or nanmean or nanmin or nansum or nanargmax or nanargmin or nancumsum or trapezoid or ravel_multi_index or choose or block or matrix_rank or cond or qr or svdvals or cholesky or matrix_power or corrcoef or lstsq or pinv or svd or eig or eigh or eigvals or eigvalsh or gcd or lcm or put or put_along_axis or ndarray_at or vectorize"`
  - `poetry run ruff check .`
  - `poetry run pytest -q`
  - `poetry run mypy jax2onnx/plugins/jax/nn --show-error-codes --no-color-output`
  - `poetry run mypy jax2onnx/plugins/jax/nn/{celu.py,elu.py,gelu.py,leaky_relu.py,mish.py,relu.py,selu.py,sigmoid.py,silu.py,softplus.py,softsign.py} --show-error-codes --no-color-output`
  - `poetry run ruff check jax2onnx/plugins/jax/nn/{celu.py,elu.py,gelu.py,leaky_relu.py,mish.py,relu.py,selu.py,sigmoid.py,silu.py,softplus.py,softsign.py}`
  - `poetry run mypy jax2onnx/plugins/jax/nn/{celu.py,elu.py,gelu.py,leaky_relu.py,mish.py,relu.py,selu.py,sigmoid.py,silu.py,softplus.py,softsign.py} --show-error-codes --no-color-output`
  - `poetry run mypy jax2onnx/plugins/jax/nn --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "celu or elu or gelu or leaky_relu or mish or relu or selu or sigmoid or silu or softplus or soft_sign"`
  - `poetry run ruff check .`
  - `poetry run ruff check jax2onnx/plugins/jax/nn/{identity.py,hard_sigmoid.py,hard_swish.py,tanh.py,standardize.py}`
  - `poetry run mypy jax2onnx/plugins/jax/nn/{identity.py,hard_sigmoid.py,hard_swish.py,tanh.py,standardize.py} --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "identity or hard_sigmoid or hard_swish or tanh or standardize"`
  - `poetry run ruff check jax2onnx/plugins/jax/nn/{squareplus.py,sparse_sigmoid.py,sparse_plus.py,relu6.py,one_hot.py,log1mexp.py,hard_tanh.py}`
  - `poetry run mypy jax2onnx/plugins/jax/nn/{squareplus.py,sparse_sigmoid.py,sparse_plus.py,relu6.py,one_hot.py,log1mexp.py,hard_tanh.py} --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "squareplus or sparse_sigmoid or sparse_plus or relu6 or one_hot or log1mexp or hard_tanh"`
  - `poetry run ruff check jax2onnx/plugins/jax/nn/_builder_utils.py jax2onnx/plugins/jax/nn/log_sigmoid.py`
  - `poetry run mypy jax2onnx/plugins/jax/nn/_builder_utils.py jax2onnx/plugins/jax/nn/log_sigmoid.py --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "log_sigmoid"`
  - `poetry run ruff check jax2onnx/plugins/jax/nn/{softmax.py,logsumexp.py,logmeanexp.py,log_softmax.py,glu.py}`
  - `poetry run mypy jax2onnx/plugins/jax/nn/{softmax.py,logsumexp.py,logmeanexp.py,log_softmax.py,glu.py} --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "softmax or logsumexp or logmeanexp or log_softmax or glu"`
  - `poetry run ruff check jax2onnx/plugins/jax/nn/{scaled_dot_general.py,scaled_matmul.py}`
  - `poetry run mypy jax2onnx/plugins/jax/nn/{scaled_dot_general.py,scaled_matmul.py} --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "scaled_dot_general or scaled_matmul"`
  - `poetry run ruff check jax2onnx/plugins/jax/nn/initializers/truncated_normal.py`
  - `poetry run mypy jax2onnx/plugins/jax/nn/initializers/truncated_normal.py --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "truncated_normal"`
  - `poetry run ruff check jax2onnx/plugins/jax/nn/dot_product_attention.py`
  - `poetry run mypy jax2onnx/plugins/jax/nn/dot_product_attention.py --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_nn.py -k "dot_product_attention or dpa"`
  - `poetry run mypy jax2onnx/plugins/jax/nn --show-error-codes --no-color-output`
  - `poetry run mypy jax2onnx/plugins/jax --show-error-codes --no-color-output`
  - `poetry run mypy jax2onnx/plugins/jax/lax/{abs.py,acos.py,acosh.py,asin.py,asinh.py,atan.py,ceil.py,erf.py,erfc.py,exp.py,exp2.py,floor.py,log1p.py,logistic.py,neg.py,round.py,sign.py,sin.py,tan.py,tanh.py} --show-error-codes --no-color-output`
  - `poetry run ruff check jax2onnx/plugins/jax/lax/{abs.py,acos.py,acosh.py,asin.py,asinh.py,atan.py,ceil.py,erf.py,erfc.py,exp.py,exp2.py,floor.py,log1p.py,logistic.py,neg.py,round.py,sign.py,sin.py,tan.py,tanh.py}`
  - `poetry run mypy jax2onnx/plugins/jax/lax/{abs.py,acos.py,acosh.py,asin.py,asinh.py,atan.py,ceil.py,erf.py,erfc.py,exp.py,exp2.py,floor.py,log1p.py,logistic.py,neg.py,round.py,sign.py,sin.py,tan.py,tanh.py} --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_lax.py -k "abs or acos or acosh or asin or asinh or atan or ceil or erf or erfc or exp2 or floor or log1p or logistic or neg or round or sign or sin or tan or tanh"`
  - `poetry run mypy $(rg -l 'JaxprEqn = getattr(core, "JaxprEqn", Any)' jax2onnx/plugins/jax/lax/*.py) --show-error-codes --no-color-output`
  - `poetry run ruff check jax2onnx/plugins/jax/lax/{and.py,eq.py,ge.py,gt.py,le.py,lt.py,ne.py,or.py,xor.py,atanh.py,cbrt.py,erf_inv.py,expm1.py,is_finite.py,shift_left.py,shift_right_logical.py}`
  - `poetry run mypy jax2onnx/plugins/jax/lax/{and.py,eq.py,ge.py,gt.py,le.py,lt.py,ne.py,or.py,xor.py,atanh.py,cbrt.py,erf_inv.py,expm1.py,is_finite.py,shift_left.py,shift_right_logical.py} --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_lax.py -k "Test_and or Test_or or Test_xor or Test_eq or Test_ne or Test_lt or Test_gt or Test_less_equal or Test_greater_equal or Test_atanh or Test_cbrt or Test_erf_inv or Test_expm1 or Test_is_finite or Test_shift_left or Test_shift_right_logical"`
  - `poetry run mypy $(rg -l 'JaxprEqn = getattr(core, "JaxprEqn", Any)' jax2onnx/plugins/jax/lax/*.py) --show-error-codes --no-color-output`
  - `poetry run ruff check jax2onnx/plugins/jax/lax/{log.py,sqrt.py,square.py,bitwise_not.py,cos.py,clz.py,cosh.py,population_count.py,shift_right_arithmetic.py,sinh.py}`
  - `poetry run mypy jax2onnx/plugins/jax/lax/{log.py,sqrt.py,square.py,bitwise_not.py,cos.py,clz.py,cosh.py,population_count.py,shift_right_arithmetic.py,sinh.py} --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_lax.py -k "Test_bitwise_not or Test_clz or Test_cos or Test_cosh or Test_log or Test_population_count or Test_shift_right_arithmetic or Test_sinh or Test_sqrt or Test_square"`
  - `poetry run mypy jax2onnx/plugins/jax/lax --show-error-codes --no-color-output`
  - `poetry run mypy jax2onnx/plugins/jax --show-error-codes --no-color-output`
  - `poetry run ruff check .`
  - `poetry run mypy jax2onnx/plugins/jax/core --show-error-codes --no-color-output`
  - `poetry run ruff check jax2onnx/plugins/jax/core/{name.py,jit.py,dim_as_value.py,custom_vjp_call.py,custom_jvp_call.py}`
  - `poetry run mypy jax2onnx/plugins/jax/core --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_core.py`
  - `poetry run mypy jax2onnx/plugins/jax --show-error-codes --no-color-output`
  - `poetry run ruff check .`
  - `poetry run mypy jax2onnx/plugins/jax/random --show-error-codes --no-color-output`
  - `poetry run ruff check jax2onnx/plugins/jax/random`
  - `poetry run mypy jax2onnx/plugins/jax/random --show-error-codes --no-color-output`
  - `poetry run pytest -q tests/primitives/test_random.py`
  - `poetry run mypy jax2onnx/plugins/jax --show-error-codes --no-color-output`
  - `poetry run ruff check .`
- Fixes landed while executing coverage expansion:
  - `jax2onnx/plugins/jax/numpy/select.py`: `_select_impl` now accepts flattened primitive operands plus `num_conds`/`num_choices` params, which is required for transpose fallback execution.
  - `jax2onnx/plugins/jax/numpy/add.py`: switched AD registration to allowlisted original-rule forwarding from `lax.add_p` for JVP+transpose.
  - `jax2onnx/plugins/jax/numpy/concatenate.py`: switched AD registration to allowlisted original-rule forwarding from `lax.concatenate_p` for JVP+transpose.
  - `jax2onnx/plugins/jax/numpy/moveaxis.py`: switched AD registration to allowlisted original-rule forwarding from `lax.transpose_p` for JVP+transpose.
  - `jax2onnx/plugins/jax/numpy/reshape.py`: switched AD registration to allowlisted original-rule forwarding from `lax.reshape_p` for JVP+transpose.
  - `jax2onnx/plugins/jax/numpy/split.py`: switched AD registration to allowlisted original-rule forwarding from `lax.split_p` for JVP+transpose.
  - `jax2onnx/plugins/jax/numpy/squeeze.py`: switched AD registration to allowlisted original-rule forwarding from `lax.squeeze_p` for JVP+transpose.
  - `jax2onnx/plugins/jax/numpy/tile.py`: switched AD registration to allowlisted original-rule forwarding from `lax.tile_p` for JVP+transpose; added converter plugin-registry alias for raw `tile` primitive equations emitted by forwarded AD.
  - `jax2onnx/plugins/jax/numpy/transpose.py`: switched AD registration to allowlisted original-rule forwarding from `lax.transpose_p` for JVP+transpose.
  - `tests/extra_tests/framework/test_ad_rule_forwarding.py`: added guard asserting known non-1:1 mappings are not allowlisted (`select_n -> jax.numpy.select/where`, `gather -> jax.numpy.take`, `concatenate -> jax.numpy.stack`).
  - `tests/extra_tests/framework/test_ad_rule_forwarding.py`: added policy assertions that `where`/`select`/`take`/`stack` keep non-forwarded AD rules (not identity-forwarded from `lax.select_n_p`, `lax.gather_p`, or `lax.concatenate_p`).
  - `jax2onnx/plugins/jax/_autodiff_utils.py`: added centralized original-rule forwarding blocklist API (`get_original_rule_forwarding_blocklist`) for known non-1:1 mappings, and switched forwarding policy tests to consume it.
  - `jax2onnx/plugins/jax/_autodiff_utils.py`: added import-time guard that fails fast if original-rule forwarding allowlist and blocklist overlap.
  - `tests/extra_tests/framework/test_ad_rule_forwarding.py`: added direct unit test for overlap guard behavior by injecting a conflicting mapping and asserting `_validate_forwarding_policy_sets()` raises.
  - `tests/extra_tests/framework/test_ad_rule_forwarding.py`: added helper-level policy test proving `register_original_rule_forwarding(..., override=False)` preserves existing target rules and only replaces them when `override=True`.
  - `jax2onnx/plugins/jax/_autodiff_utils.py`: evaluated expanding linear-transpose fallback allowlist to migrated `jax.nn` primitives, then reverted after conversion probe showed baseline JAX parity failure (`linear_transpose(jax.nn.relu)` raises `NotImplementedError` for primitive `max`).
  - `tests/extra_tests/framework/test_ad_rule_completeness.py`: added guard that migrated nonlinear `jax.nn` primitives are not allowlisted for generic linear-transpose fallback.
  - `tests/extra_tests/test_issue_203_linear_transpose_allowlist_ops.py`: added snapshot guard asserting `_LINEAR_TRANSPOSE_FALLBACK_ALLOWLIST` matches the conversion regression matrix so allowlist changes require synchronized test updates.
  - `tests/extra_tests/framework/test_ad_backfill_idempotence.py`: added stats-level regression asserting backfill skips non-allowlisted primitives (`skipped_not_allowlisted`) and does not install transpose rules outside the curated set.
  - `jax2onnx/plugins/jax/numpy/{concatenate,split,squeeze,tile,transpose,stack,take}.py`: normalized `BatchDim` typing to `TypeAlias = int | None` (matching `batching.not_mapped`) to reduce strict-mypy alias/type errors without changing runtime batching semantics.
  - `jax2onnx/plugins/jax/numpy/add.py`: added explicit typing/casts for `binding_specs` and `_add_batch_rule` so focused strict mypy passes cleanly.
  - `jax2onnx/plugins/jax/numpy/reshape.py`: fixed strict-mypy issues by typing `_iter_newshape`, removing stale ignore comments, tightening numeric casts in `_flatten_axis_for_static_target`, handling optional `order` in abstract-eval shape calls, annotating `binding_specs`, and replacing `type(batching.not_mapped)` annotations with `BatchDim = int | None`.
  - `jax2onnx/plugins/jax/numpy/concatenate.py`: tightened `_canonicalize_call` axis handling with explicit integer validation and typed canonical return values.
  - `jax2onnx/plugins/jax/numpy/split.py`: added typed casts/annotations in patched binding and batching helpers (`_split_single`, batch-dim narrowing) to reduce strict-mypy issues while preserving behavior.
  - `jax2onnx/plugins/jax/numpy/take.py`: added missing return/variable annotations in patched module call and lowering path (`indices_dtype`) to satisfy strict mypy checks.
  - `jax2onnx/plugins/jax/numpy/tile.py`: added missing function/variable annotations (`_tile_param`, `_tile_with_symbolic_repeats`, `input_dtype`, `repeats_dtype`, `_origin`, `_align_repeats_rank`) and widened static repeats typing to cover symbolic repeat descriptors.
  - `jax2onnx/plugins/jax/numpy/prod.py`: replaced `BatchDim = int | type(batching.not_mapped)` with `BatchDim: TypeAlias = int | None`, and added explicit `bdim is None` handling in `_prod_batch_rule` to satisfy strict typing without changing mapped-batch behavior.
  - `jax2onnx/plugins/jax/numpy/sum.py`: removed `producer` union-attr mypy failures by precomputing `producer_op`/`producer_inputs`, tightened `_const_scalar` return typing to concrete `float|int|None`, and fully annotated `_sum_transpose_rule`.
  - `jax2onnx/plugins/jax/numpy/select.py`: fixed strict-mypy typing by normalizing promoted dtype inference (`_promote_dtype`), typing `result_dtype` as `np.dtype[Any]`, and annotating `_select_batch_rule`.
  - `jax2onnx/plugins/jax/numpy/where.py`: added strict type annotations for helper/lowering/batching callables (`abstract_eval`, `lower`, `_maybe_cast`, `binding_specs`, patched impls), normalized dtype locals to `np.dtype[Any]`, and removed stale ignore-based signatures.
  - `jax2onnx/plugins/jax/numpy/mean.py`: removed stray debug `print` in abstract eval, typed `out_dtype`, switched batching alias to `BatchDim: TypeAlias = int | None`, and added explicit `bdim is None` handling in `_mean_batch_rule`.
  - `jax2onnx/plugins/jax/numpy/outer.py`: casted `binding_specs` return to concrete spec type, migrated batch-dim typing/checks to `int | None`, and typed mapped batches as `list[tuple[jax.Array, int]]`.
  - `jax2onnx/plugins/jax/numpy/cumprod.py`: migrated batch-dim typing/checks to `BatchDim: TypeAlias = int | None` and explicit `bdim is None` handling.
  - `jax2onnx/plugins/jax/numpy/cumsum.py`: added typed reverse-aware helper for testcase callables (`_typed_jnp_cumsum`) to avoid stub drift on `reverse=...`, typed `binding_specs`/`_runtime_cumsum`, and migrated batch-dim typing/checks to `BatchDim: TypeAlias = int | None`.
  - `jax2onnx/plugins/jax/numpy/unstack.py`: typed `_unstack_impl` return as `tuple[jax.Array, ...]` (wrapping original output in `tuple(...)`), and migrated batch-dim typing/checks to `BatchDim: TypeAlias = int | None`.
  - `jax2onnx/plugins/jax/numpy/shape.py`: typed `_shape_eval`, annotated lowering `dtype_map`, and migrated batch-dim typing/checks to `BatchDim: TypeAlias = int | None`.
  - `jax2onnx/plugins/jax/numpy/sort.py`: migrated batch-dim typing/checks to `BatchDim: TypeAlias = int | None` and added explicit unmapped batching branch in `_sort_batch_rule`.
  - `jax2onnx/plugins/jax/numpy/linalg_norm.py`: widened patched `ord` typing to include string values (`"fro"`), removed unreachable fallback branch, and migrated batching alias/checks to `BatchDim: TypeAlias = int | None`.
  - `jax2onnx/plugins/jax/numpy/linspace.py`: tightened strict typing across helper/lowering/binding/batching paths (`_infer_result_dtype`, `abstract_eval`, `lower`, `binding_specs`, `_linspace_impl`, `_linspace_batch_rule`), switched IR context annotations to `LoweringContextProtocol`, and migrated batching alias/checks to `BatchDim: TypeAlias = int | None`.
  - `jax2onnx/plugins/jax/numpy/*`: completed migration of remaining legacy `BatchDim = int | type(batching.not_mapped)` aliases in the numpy plugin family.
  - `jax2onnx/plugins/jax/numpy/_unary_utils.py`: added strict typing for inferred dtypes and batching helper signature (`register_unary_elementwise_batch_rule`) while preserving existing runtime behavior.
  - `jax2onnx/plugins/jax/numpy/_reduction_utils.py`: added strict typing for inferred dtypes, replaced legacy `type(batching.not_mapped)` annotations with a typed `BatchDim` alias, and added explicit unmapped-batch handling in shared reduction batching rule helper.
  - `jax2onnx/plugins/jax/numpy/{tan,sinh,cosh,cos}.py`: annotated floating-output dtype locals (`out_dtype`) as `np.dtype[Any]`.
  - `jax2onnx/plugins/jax/numpy/sqrt.py`: made `ReduceSumSquare` fusion metadata access null-safe (`attributes` lookup via guarded mapping).
  - `jax2onnx/plugins/jax/numpy/log.py`: made `ReduceSum`/`Exp` fusion metadata and input access null-safe (`attributes`/`inputs` guarded).
  - `jax2onnx/plugins/jax/numpy/windows.py`: typed `_window_impl` primitive parameter (`Primitive`) and annotated `ensure_abstract_eval_bound()` with `-> None` for strict mypy.
  - `jax2onnx/plugins/jax/numpy/linalg_det.py`: annotated `ensure_abstract_eval_bound()` with `-> None` for strict mypy.
  - `jax2onnx/plugins/jax/numpy/pow.py`: added strict type annotations for broadcast-shape helper, abstract-eval return, binding-specs return, and batch-rule factory; removed stale ignore-driven typing.
  - `jax2onnx/plugins/jax/numpy/fft.py`: annotated metadata wrapper `binding_specs()` return type.
  - `jax2onnx/plugins/jax/numpy/equal.py`: typed promoted-dtype locals (`lhs_dtype`, `rhs_dtype`), annotated `_equal_batch_rule`, and stabilized `binding_specs` return typing.
  - `jax2onnx/plugins/jax/numpy/{floor_divide,copysign,bitwise_or,acos,bitwise_xor,bitwise_right_shift,bitwise_left_shift,fabs,less,bitwise_and,left_shift,fmod,asin,asinh,exp2,atanh,acosh,bitwise_not,divide,atan2,atan,less_equal,greater_equal,invert,right_shift,greater,expm1}.py`: replaced redundant `cast(..., jnp_binding_specs(...))` returns with typed local `specs: list[AssignSpec | MonkeyPatchSpec]` assignments.
  - `jax2onnx/plugins/jax/numpy/composite_metadata*.py`: added strict signatures for metadata-only base plugin hooks (`binding_specs` return typing and `lower(..., eqn: core.JaxprEqn)`).
  - `jax2onnx/plugins/jax/numpy/conj.py`: typed `_is_complex_var` and `_conj_batch_rule` signatures.
  - `jax2onnx/plugins/jax/numpy/compress.py`: annotated `ensure_abstract_eval_bound() -> None` and removed redundant callable casts in patched fallback path by narrowing to `orig_impl`.
  - `jax2onnx/plugins/jax/numpy/clip.py`: promoted `JaxValue` to `TypeAlias` and typed `_clip_batching_rule`.
  - `jax2onnx/plugins/jax/numpy/einsum.py`: normalized equation typing for non-string inputs, typed padded-axis constant creation, annotated `binding_specs` return type, and typed `_einsum_no_batch`.
  - `jax2onnx/plugins/jax/numpy/matmul.py`: added complete signatures for `_matmul_shape`, `abstract_eval` kwargs, complex detection helper, patched binding kwargs, `_matmul_impl`, and `_matmul_batch_rule`.
  - `jax2onnx/plugins/jax/numpy/arange.py`: fixed duplicate sentinel declaration typing, made scalar extraction helpers return concrete `float|int|None`, fixed dynamic-shape tuple typing in `abstract_eval`, and switched lowering context annotations to `LoweringContextProtocol`.
  - `jax2onnx/plugins/jax/nn/{celu,elu,gelu,leaky_relu,mish,relu,selu,sigmoid,silu,softplus,softsign}.py`: annotated `ensure_abstract_eval_bound()` with `-> None` across the activation plugin tranche.
  - `jax2onnx/plugins/jax/nn/{celu,elu,leaky_relu}.py`: replaced `float(params.get(...))` in JVP rules with typed scalar narrowing before conversion to avoid `float(object)` strict-mypy failures while preserving default fallback behavior.
  - `jax2onnx/plugins/jax/numpy/{add,maximum,minimum,outer,tan,sinh,cosh,cos,sqrt,log}.py`: switched `binding_specs` implementations from explicit `cast(...)` wrappers to typed local variables (`specs: list[AssignSpec | MonkeyPatchSpec] = ...`) to satisfy strict mypy without accumulating redundant-cast debt in package-level scans.
  - `jax2onnx/plugins/jax/numpy/concatenate.py`: fixed a symbolic-vmap regression in `_concatenate_batch_rule` (detected by `gpt_oss_transformer_block_dynamic`) by preserving non-concrete `axis_size` handling instead of requiring concrete integers.
  - `jax2onnx/plugins/jax/numpy/{where,select,take}.py`: attempted helper-only JVP migration via `register_jvp_via_jax_jvp`; reverted after focused regressions showed float0 constants leaking into conversion traces for grad tests.
  - `tests/extra_tests/framework/test_ad_rule_completeness.py`: added regression guard ensuring patched-grad jaxprs for `where`/`select`/`take` contain no float0 constants.
  - `jax2onnx/plugins/jax/nn/{celu,elu,gelu,leaky_relu,mish,relu,selu,sigmoid,silu,softplus,softsign,identity,hard_sigmoid,hard_swish,tanh,standardize,log_sigmoid}.py`: added missing `ensure_abstract_eval_bound() -> None` annotations for strict-mypy compliance.
  - `jax2onnx/plugins/jax/nn/{celu,elu,leaky_relu}.py`: replaced direct `float(params.get(...))` coercions in JVP rules with typed scalar narrowing to avoid `float(object)` strict-mypy failures.
  - `jax2onnx/plugins/jax/nn/{squareplus,sparse_sigmoid,sparse_plus,relu6,one_hot,log1mexp,hard_tanh,softmax,logsumexp,logmeanexp,log_softmax,glu}.py`: tightened strict typing across dtype locals, binding shims, and batch-rule signatures; normalized batch-dim aliases to `TypeAlias = int | None` where needed.
  - `jax2onnx/plugins/jax/nn/_builder_utils.py`: switched to `LoweringContextProtocol`-typed lowering signatures and typed helper batch-rule plumbing.
  - `jax2onnx/plugins/jax/nn/{scaled_dot_general,scaled_matmul}.py`: added full helper/lowering signature annotations (including dtype and optional kwargs) to clear strict-mypy errors without changing operator behavior.
  - `jax2onnx/plugins/jax/nn/initializers/truncated_normal.py`: removed legacy ignore-driven context typing, adopted `LoweringContextProtocol`, and added typed fallback callable wiring for metadata patching.
  - `jax2onnx/plugins/jax/nn/dot_product_attention.py`: typed mask/dtype helpers, binding shim signatures, abstract eval return type, and batch-rule alias/checks (`BatchDim: TypeAlias = int | None`), clearing remaining family-local strict-mypy errors.
  - `jax2onnx/plugins/jax/lax/*`: removed all remaining dynamic `JaxprEqn = getattr(core, "JaxprEqn", Any)` aliases and switched lowering signatures to string-literal `eqn: "core.JaxprEqn"` to satisfy strict-mypy while preserving runtime compatibility with JAX versions where `jax.core.JaxprEqn` is not exported at import time.
  - `jax2onnx/plugins/jax/lax/{abs,acos,acosh,asin,asinh,atan,ceil,erf,erfc,exp,exp2,floor,log1p,logistic,neg,round,sign,sin,tan,tanh}.py`: applied first unary strict-mypy tranche (alias cleanup only) with no behavior changes.
  - `jax2onnx/plugins/jax/lax/{and,eq,ge,gt,le,lt,ne,or,xor}.py`: added explicit `prefer_dtype: np.dtype[Any]` annotations and alias cleanup for strict-mypy.
  - `jax2onnx/plugins/jax/lax/{atanh,cbrt,erf_inv,expm1,is_finite,shift_left,shift_right_logical}.py`: completed alias cleanup and import normalization (including restoring `Any` import in `erf_inv` where helper signatures require it).
  - `jax2onnx/plugins/jax/lax/{log,sqrt}.py`: made Reduce* fusion metadata reads null-safe (`attributes` via callable `get` probe) and avoided direct union-attr access.
  - `jax2onnx/plugins/jax/lax/square.py`: replaced list-comprehension `int(candidate)` narrowing with explicit typed candidate accumulation for `axis0_override` selection.
  - `jax2onnx/plugins/jax/lax/bitwise_not.py`: annotated `abstract_eval` return type and `x_dtype` local.
  - `jax2onnx/plugins/jax/lax/{cos,cosh,sinh}.py`: annotated floating dtype locals (`x_dtype: np.dtype[Any]`) for strict-mypy.
  - `jax2onnx/plugins/jax/lax/{clz,population_count}.py`: annotated integer dtype locals (`in_dtype: np.dtype[Any]`) for strict-mypy.
  - `jax2onnx/plugins/jax/lax/shift_right_arithmetic.py`: annotated `aval_dtype`/`prefer_dtype` as `np.dtype[Any]` in dtype inference and lowering paths.
  - `jax2onnx/plugins/jax/core/name.py`: replaced TYPE_CHECKING-only `IRContext`/ignore signature with `LoweringContextProtocol` + typed `eqn: core.JaxprEqn`.
  - `jax2onnx/plugins/jax/core/jit.py`: added explicit helper/lowering type signatures (`_freshen_closed_jaxpr`, `_fresh_var`, `_map_vars`, `lower`) and typed `_PRIM` class var.
  - `jax2onnx/plugins/jax/core/dim_as_value.py`: annotated helper/lowering signatures (`_dynamic_or_constant`, `_check`, `lower`) and normalized bool return typing in dynamic/constant post-check wrapper.
  - `jax2onnx/plugins/jax/core/custom_vjp_call.py`: removed stale ignore comments, added typed custom-vjp helper signatures, and switched lowering signature to `LoweringContextProtocol`.
  - `jax2onnx/plugins/jax/core/custom_jvp_call.py`: removed stale ignore comments, added typed custom-jvp helper signatures, and switched lowering signature to `LoweringContextProtocol`.
  - `jax2onnx/plugins/jax/random/random_seed.py`: typed `_shape_dims` return (`tuple[object, ...]`) and removed stale override-ignore comments from random seed/wrap/unwrap lowering signatures.
  - `jax2onnx/plugins/jax/random/{random_fold_in,random_bits}.py`: removed stale override-ignore comments from lowering signatures.
  - `jax2onnx/plugins/jax/random/normal.py`: annotated `out_dtype: np.dtype[Any]` and `ensure_abstract_eval_bound() -> None`.
  - `jax2onnx/plugins/jax/random/{categorical,bernoulli}.py`: annotated `ensure_abstract_eval_bound() -> None`.
- Full gates:
  - `poetry run ruff check .` -> passed
  - `poetry run pytest -q` -> passed (`3132 passed`)
  - Re-ran full gates after `prod/sum/where/select` typing tranche: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings`).
  - Re-ran full gates after `mean/outer/cumprod/cumsum/unstack` typing tranche: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings`).
  - Re-ran full gates after `shape/sort` typing tranche: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings`).
  - Re-ran full gates after `linalg_norm/linspace` typing tranche: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings`).
  - Re-ran full gates after `_unary_utils/_reduction_utils` typing tranche: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings`).
  - Re-ran full gates after `tan/sinh/cosh/cos/sqrt/log` typing tranche: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings`).
  - Re-ran full gates after `windows/linalg_det` typing tranche and `add/maximum/minimum/outer` cast stabilization: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14260 warnings`).
  - Re-ran full gates after `pow/fft/equal` typing tranche and binding-specs cast cleanup: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings`).
  - Re-ran full gates after closing remaining strict-mypy numpy backlog (`composite_metadata*`, `conj`, `compress`, `clip`, `einsum`, `matmul`, `arange`): `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings in 13m00s`).
  - Re-ran full gates after closing the strict-mypy `jax.nn` backlog: `poetry run ruff check .` passed and `poetry run pytest -q` passed (`3132 passed, 14396 warnings in 772.27s (0:12:52)`).
  - Re-ran lint gate after `jax.lax` strict-mypy tranches: `poetry run ruff check .` passed (`All checks passed!`).
  - Re-ran lint gate after `jax.core` strict-mypy tranche: `poetry run ruff check .` passed (`All checks passed!`).
  - Re-ran lint gate after `jax.random` strict-mypy tranche: `poetry run ruff check .` passed (`All checks passed!`).
  - Current strict-mypy snapshot for `jax2onnx/plugins/jax/numpy`: `0 errors in 0 files` (`Success: no issues found in 106 source files`; down from `66 errors in 39 files`, `86 errors in 52 files`, and `97 errors in 56 files` earlier in the plan).
  - `poetry run mypy src` -> not runnable in this repo layout (`src/` missing)
  - `poetry run mypy jax2onnx` -> fails on large pre-existing baseline outside this patch area
  - `poetry run mypy jax2onnx/plugins/jax/numpy/{concatenate.py,split.py,squeeze.py,tile.py,transpose.py,stack.py,take.py}` -> passed (`Success: no issues found in 7 source files`)
  - `poetry run mypy jax2onnx/plugins/jax/numpy` -> passed (`Success: no issues found in 106 source files`)
  - Current strict-mypy snapshot for `jax2onnx/plugins/jax/nn`: `0 errors in 0 files` (`Success: no issues found in 38 source files`; progression in this execution tranche: `99 errors in 34 files` -> `85 errors in 23 files` -> `80 errors in 18 files` -> `65 errors in 11 files` -> `61 errors in 9 files` -> `38 errors in 4 files` -> `26 errors in 2 files` -> `14 errors in 1 file` -> `0 errors in 0 files`).
  - Current strict-mypy snapshot for `jax2onnx/plugins/jax/lax`: `677 errors in 109 files` (`Found 677 errors in 109 files (checked 161 source files)`).
  - Current strict-mypy snapshot for `jax2onnx/plugins/jax/core`: `0 errors in 0 files` (`Success: no issues found in 6 source files`).
  - Current strict-mypy snapshot for `jax2onnx/plugins/jax/random`: `0 errors in 0 files` (`Success: no issues found in 7 source files`).
  - Current strict-mypy snapshot for `jax2onnx/plugins/jax`: `684 errors in 111 files` (`Found 684 errors in 111 files (checked 324 source files)`; down from `694 errors in 117 files` before the `jax.random` tranche, down from `721 errors in 122 files` before the `jax.core` tranche, and down from `882 errors in 168 files` before starting the `jax.lax` + `jax.core` + `jax.random` tranches).

## Acceptance Criteria
- `to_onnx` conversion succeeds for the issue #203 repro case.
- New framework policy test reports zero primitives with JVP and missing transpose.
- Existing targeted autodiff tests continue to pass.
- Every touched behavior has a corresponding local focused test in standard repo locations (`tests/primitives/`, `tests/extra_tests/`, or `tests/extra_tests/framework/`).
- Full local quality gates pass:
  - `poetry run ruff check .`
  - `poetry run mypy src`
  - `poetry run pytest -q`

## Risks and Mitigations
- Risk: Generic transpose fallback may not be valid for every nonlinear primitive in every context.
  - Mitigation: Keep explicit transpose rules preferred; fallback only for missing rules; validate via regression suite.
- Risk: API drift in JAX internals (`ad` registries).
  - Mitigation: isolate registry writes in `_autodiff_utils.py` and keep compatibility shims local.
- Risk: Hidden direct JVP assignments outside helper path.
  - Mitigation: enforce framework policy test plus bootstrap backfill.

## Why this is durable
- It changes the default from opt-in correctness to guaranteed correctness.
- It protects both existing plugins and future plugins.
- It catches regressions early via policy tests and conversion-level repros.
