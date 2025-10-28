# Merge Notes — PR #106 (`broadcast_in_dim_fix`)

## Current Situation
- Rebasing/merging the PR onto `main` hits conflicts in seven files:
  - `jax2onnx/converter/ir_constants.py`
  - `jax2onnx/converter/ir_context.py`
  - `jax2onnx/converter/lower_dimexpr.py`
  - `jax2onnx/plugins/jax/lax/broadcast_in_dim.py`
  - `jax2onnx/plugins/jax/lax/gather.py`
  - `jax2onnx/plugins/jax/lax/gather_compile.py`
  - `jax2onnx/plugins/jax/lax/gather_helpers.py`
- Root cause: the PR carries an older constant-folding API (`try_evaluate_const(var, handler)`) and slimmer lowering codepaths, while `main` now ships richer constant-folder registration, stack-trace metadata, loop extent metadata, and a reworked gather/broadcast stack. We need to adopt the newer infrastructure without losing the fixes introduced in the PR.

## Resolution Strategy

### 1. `ir_constants.py` + `ir_context.py`
- **Baseline**: take `origin/main` versions. They introduce `ConstantFolder.register_handler`, caching, and stack-trace metadata plumbing.
- **Reapply PR intent**:
  1. Verify every callsite that still does `ctx.try_evaluate_const(var, handler)` is updated to rely on handler registration (`ctx.register_constant_evaluator`) exactly as `main` expects. The `gather` plugin already wires this up in `main`; extend the same registration list if the PR needs additional primitives.
  2. Confirm any new helpers from the PR still receive deterministic evaluation (add registrations if broadcasting relies on them).
- **Side quest**: ensure `IRContext.try_evaluate_const` continues returning `NDArray` (the PR’s callers assume `np.ndarray`), but keep the centralized handler cache from `main`.

### 2. `lower_dimexpr.py`
- **Baseline**: start from `origin/main` so we inherit the newer `_DimExpr` lowering semantics.
- **Port PR improvements**:
  - Keep the PR’s ability to accept `ir.Value` inputs in `LowerDimExpr.__call__`.
  - Preserve the enhanced `_set_metadata` that stamps `(len(values),)` instead of `(1,)` for concatenated vectors.
  - Ensure the cache keys include values passed straight through (avoid double-lowering).
  - Run `pytest tests/extra_tests/loop/test_loop_broadcast_extent_regression.py` afterwards; that testset exercises the new symbolic extent coverage.

### 3. `broadcast_in_dim.py`
- **Baseline**: retain the newer `main` file because it has:
  - Axis-0 override propagation (`ensure_axis0_extent`, loop metadata).
  - Extra constant folding paths (`_materialize_constant_array`, `CastLike` handling).
- **Reintroduce PR fix**:
  1. Merge the PR’s `modified_target_shape` idea: prefer loop scatter hints when possible, and skip a redundant `Expand` when `reshape_dims` already matches.
  2. Keep `_maybe_inline_constant_broadcast` inline evaluation but ensure it now calls `ctx.try_evaluate_const` without the handler argument.
  3. Validate both static and symbolic paths with `pytest tests/extra_tests/loop/test_loop_broadcast_extent_regression.py::test_loop_concat_extent_regression`.

### 4. `gather.py`
- **Baseline**: stick with the expanded `main` implementation (`_ensure_constant_folders_registered`, `GatherND` detection, richer GIR support).
- **Carry PR intent**:
  - Make sure any helper the PR relied on (e.g., simplified GIR conversions) is still reachable—most are already present in `main`.
  - Remove the now-obsolete `_eval_primitive` helper after swapping callers to the registered constant evaluators.
  - Re-run the new regression tests from `main` plus PR-specific ones: `pytest tests/extra_tests/test_gather_modes.py tests/extra_tests/framework/test_do_not_skip_numeric_validation.py`.

### 5. `gather_compile.py` & `gather_helpers.py`
- **Baseline**: use the structured `main` versions (explicit imports, typed helpers).
- **Integrate PR deltas**:
  - Compare algorithms for GIR normalization; the PR may have minor differences (e.g., extra index reshapes). Port any missing logic rather than wholesale replacing the file.
  - Ensure helper signatures remain stable for callers in `gather.py`.
  - Confirm constant folding of index tensors still functions after merging (`pytest tests/extra_tests/test_gather_modes.py` exercises this).

## Suggested Workflow
1. `git fetch origin && git checkout broadcast_in_dim_fix`
2. Create a safety branch (`git checkout -b broadcast_in_dim_fix-merge-main`) before merging.
3. `git merge origin/main` and tackle conflicts in the order above; prefer launching a merge tool for the large plugins to avoid dropping subtle metadata handling.
4. After edits, run the focused tests:
   - `poetry run pytest -q tests/extra_tests/loop/test_loop_broadcast_extent_regression.py`
   - `poetry run pytest -q tests/extra_tests/test_gather_modes.py`
   - `poetry run pytest -q tests/extra_tests/framework/test_do_not_skip_numeric_validation.py`
5. If all green, run the project checklist (`ruff`, `mypy`, full `pytest -q`) before pushing.

## Follow-ups / Validation
- Double-check docs or expect-graph fixtures for broadcast/gather; regenerate via `poetry run python scripts/emit_expect_graph.py <testcase>` if structural outputs changed.
- Keep an eye on axis-0 metadata coverage—if new hints alter the graph, add/update expect-graph snapshots.
- Once merge is clean and tests pass, consider filing a quick TODO to collapse duplicated constant-folder registration between gather and broadcast plugins if we keep extending the primitive list.
