# PR 102 – Use the graph object to track nodes and values

## PR snapshot
- Title: “Use the graph object to track nodes and values” (draft, open).
- Author: @justinchuby (collaborator); branch `enpasos:justinchu/use-graph`.
- Created: 2025-10-17; last update: 2025-10-18.
- Commits: 1 (`0923097f` “Remove some lists”); +21 / −50 LOC across 2 files.
- Goal: Drop manual list bookkeeping in `IRBuilder` by delegating to `onnx_ir.Graph` via `_TapeBuilder`; eliminate `_sync_from_tape_builder`.

## Review threads worth tracking
- General question to @enpasos about direction (remove `sync`, rely on graph) – awaiting feedback.
- Discussion about eventually copying `ir.Graph` in `to_ir_model` to avoid mutating shared state.
- Follow-up confirms approach likely extends existing inliner logic from upstream `onnx_ir`.

## Observed gaps / risks in current diff
- `IRBuilder.__init__` references `self.inputs/outputs/nodes` before they are defined; we need concrete containers or to redirect to `self.graph` helpers.
- Property `initializers` now exposes `MutableMapping`; membership checks (`weight in builder.initializers`) will fail unless we wrap/adapter.
- Function-mode path still appends to `self.nodes`, which should now route through the graph’s node list.
- `add_initializer_from_scalar` early-return for existing name assumes `graph.initializers` behaves like dict; verify replacement semantics & dtype handling.
- `to_ir_model` used to reuse `self.graph` directly; keep verifying the new `clone_graph` helper preserves metadata/attrs during export.
- Broader test coverage limited to `tests/extra_tests/converter/test_ir_builder.py`; we should add regression for tape-builder paths and ensure expect_graph fixtures stay in sync.

## Plan of attack
1. **Reconcile IRBuilder containers**: audit `onnx_ir.Graph` API, ensure `inputs`, `outputs`, `nodes`, `initializers` are backed by shared live lists/maps; initialize builder fields safely before creating the graph.
2. **Clean up builder helpers**: route all node/initializer writes through the graph/tape builder, remove stale sync code, and adapt function-mode path so it still records nodes correctly.
3. **Handle model export immutability**: explore `ir.Graph.copy()` or manual clone so `to_ir_model` returns a fresh graph/model without aliasing `IRBuilder.graph`.
4. **Tighten tests**: extend `tests/extra_tests/converter/test_ir_builder.py` (and any expect_graph fixtures) to cover initializer overwrite, tape-builder operations, and repeated `to_ir_model` calls.
5. **Verification**: run focused pytest (`poetry run pytest -q tests/extra_tests/converter/test_ir_builder.py`) followed by lint/mypy (`poetry run ruff check .`, `poetry run mypy src`) once implementation stabilizes.

## Notes 2025-10-19
- Added `jax2onnx/converter/ir_clone.py` with a local `clone_graph` helper modeled after `_CopyReplace.clone_graph` and the discussion in onnx/ir-py#172. `IRBuilder.to_ir_model` now clones before building `ir.Model`.
- New regression coverage in `tests/extra_tests/converter/test_ir_clone.py` to assert cloned graphs preserve metadata, detach values/nodes/initializers, and duplicate graph attributes used by control-flow ops.
- Rewired `IRBuilder` containers to use the live `onnx_ir.Graph` storage (`inputs`, `outputs`, node sequence) and layered an `_InitializerList` shim so plugins can continue to append/remove while the graph keeps canonical mappings. Added `test_ir_builder_initializer_view_assignment_roundtrip` and re-ran `pytest -q tests/extra_tests/converter/test_ir_builder.py tests/extra_tests/converter/test_ir_clone.py tests/extra_tests/converter/test_jaxpr_converter_interaction_with_builder.py`.
