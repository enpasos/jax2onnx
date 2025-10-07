# ONNX IR Builder Migration – Working Notes

These notes track outstanding work needed to bring the codebase in line with the guidance from `docs/dev_guides/onnx_ir_builder.md`. Tasks are grouped from highest priority (blocking correctness or policy) to longer-term cleanup. Always keep the builder guardrails in mind: emit IR via `ctx.builder`/`onnx_ir._tape.Builder`, pass `_outputs` as sequences, register constants through the builder, and stamp dtype/shape metadata on every produced value.

---

## 0. Reference Checklist (keep handy)
- ✅ Prefer builder calls over manual `ir.Node` creation; only fall back when we genuinely need legacy behaviour (e.g., Function bodies without initializer support).
- ✅ Initializers must flow through `builder.initializer(...)` / `ctx.bind_const_for_var(...)`.
- ✅ Every builder result needs dtype + shape stamped (`_stamp_type_and_shape`) and ValueInfo registration (`_ensure_value_info` / `_add_value_info`).
- ✅ `_outputs` arguments must be sequences; enforce the guardrail introduced in the builder doc.
- ✅ No protobuf (`onnx`) imports in `converter/` or `plugins/`.

---

## 1. Hot Fixes / Regressions

| Item | Status | Notes |
| --- | --- | --- |
| `cast_param_like` const fast-path | ✅ done | Constants now rewrite their dtype in-place, resolving `layer_norm_symbolic_batch_seq10_feat3_dynamic` drift. Keep an eye on other helpers that mutate `const_value`. |

---

## 2. High-Priority Builder TODOs

1. **Eliminate manual `ir.Node` fallbacks in active plugins**
     - Audit other recent migrations (`jax/numpy/split.py`, `jax/numpy/einsum.py`, `jax/numpy/cumsum.py`, `jax/lax/reshape.py`) to ensure no stale manual node paths remain (search for `ctx.add_node(`, `ir.Node(`).

2. **Builder coverage report for plugins**
   - Generate an inventory of remaining `ir.Node(` occurrences under `jax2onnx/plugins/`. Confirm the following are either legit (testing utilities) or schedule refactors:
     - `plugins/plugin_system.py` – call wiring & SSA scaffolding.
     - Flax NNX helpers (`max_pool`, `group_norm`, `rms_norm`) – still constructing Nodes by hand.
     - `jax/nn/dot_product_attention.py` – uses manual bookkeeping around attention mask rewrites.
   - For each real lowering that still emits `ir.Node`, open a TODO in this file with actionable steps.

3. **Control-flow leftovers**
   - `jax2onnx/plugins/plugin_system.py` fabricates `Call`/function wiring nodes directly. Evaluate whether new builder helpers are required or we document these as intentional exceptions.
   - `jax2onnx/converter/ir_context.py` continues to expose `add_node` for legacy paths. Short-term: leave as compatibility, but ensure all new plugin code routes through the builder first.

---

## 3. Medium-Term Refactors

1. **Normalize dtype/shape stamping helpers**
   - Several plugins still duplicate `desired_name` / dtype propagation logic. Extract a shared utility (e.g., `maybe_copy_spec_metadata(derived_val, spec)`) to reduce drift.

2. **Shared builder helpers for indexing**
   - `jax/lax/transpose.py`, `jax/numpy/linspace.py` and siblings still declare private helpers. Consider centralising common shape/value-info code in `_index_utils`.

3. **Test coverage for builder guardrails**
   - Add focused tests under `tests/extra_tests/framework/` that fail if `_outputs` is passed a string, or if manual `ir.Node` creation creeps back into hot paths.
   - Wire `scripts/check_ir_builder_usage.py` into CI if not already (verify pre-commit / GH workflows).

4. **Audit const casting helpers**
   - After improving `cast_param_like`, review all call sites (`group_norm`, `rms_norm`, Eqx/NNX layers) to ensure they honour the new behaviour and remove redundant manual casts.

---

## 4. Low-Priority / Nice-to-have

1. **Documentation backfill**
   - Update `docs/design.md` and `docs/expect_graph_reference.md` with a short section on builder expectations (linking to the dev guide).
   - Ensure plugin onboarding docs mention the single-use RNG rules and builder-first workflow.

2. **Automation**
   - Consider a `poetry run python scripts/check_ir_builder_usage.py --diff` mode for quick local validation before committing.

---

## 5. Tracking Table – Direct `ir.Node` Usage (needs triage)

All plugin lowers now rely on builder helpers. Keep spot-checking periodically (\`rg "ir.Node" jax2onnx/plugins\`) when new primitives land.

---

## 6. Validation Routine (run before closing TODO batches)
1. `poetry run python scripts/check_ir_builder_usage.py`
2. `poetry run ruff check . && poetry run ruff format --check .`
3. `poetry run pytest -q` (plus targeted suites, e.g., `tests/primitives/test_jnp.py::Test_linspace`)
4. For builder-heavy refactors, run a sanity `poetry run pytest -q tests/extra_tests/framework/test_ir_builder_contracts.py`

---

### Last updated
2025‑10‑07 — snapshot after migrating reshape/cumsum/einsum/linspace/split/transpose to builder calls and fixing layer-norm constant casting. Update this log with each new migration or policy change.***
