# PR 96 Review Notes

## Priority Tasks
- **P1 – Replace dynamic IR access with typed API usage**: The upstream `onnx_ir` package already publishes inline typing info (`py.typed` per PEP 561), so we should lean on that instead of maintaining local `.pyi` shims. Update our tooling to consume the installed package directly, then keep shrinking legacy `getattr`/try blocks so static checkers catch unsupported fields. Tighten mypy configuration gradually on `converter/` and `plugins/`, and reinforce the plan with a lint check that flags dynamic IR access. This keeps call sites routed through the official helpers (`value.producer()`, `node.attributes`, `ir.convenience.*`) and eliminates the anti-pattern.
  - Ensure `onnx_ir` is pulled from PyPI (>=0.1.11) inside the Poetry environment so both runtime code and type checkers see the same artifacts; drop `mypy_path` overrides pointing at `typings/`.
    - ✅ Removed the local `typings/onnx_ir` stubs and updated `pyproject.toml` so mypy consumes the inline types shipped with `onnx_ir`.
    - ✅ Aligned dev tooling + pre-commit to `mypy==1.15.0` via `scripts/run_mypy.py` (local hook, `pass_filenames: false`) so the config-scoped file list is respected even when IDE PATH lacks `poetry`; the scoped modules now pass cleanly.
  - Retire or vendor-check the scratch stubs in `typings/onnx_ir/`; replace them with smoke tests that fail fast if the real package slips missing attributes.
    - ✅ Added a top-level export assertion to `tests/extra_tests/framework/test_onnx_ir_surface.py` to guard against upstream attr regressions.
    - ✅ Consolidated the typing guidance under `docs/dev_guides/onnx_ir_typing_how_to.md` so future updates live with the rest of the dev docs.
  - Plan: layer per-module strictness on `converter/` + `plugins/` once we finish the cleanup sweep; start with graph/optimizer utilities after the initial inline-type adoption.
    - ✅ `jax2onnx/converter/function_scope.py` now passes mypy; adjusted `FunctionScope` guards and IR pretty-printer helpers to avoid optional name leaks while keeping override dumps readable.
    - ✅ `converter/conversion_api.py`, `converter/ir_context.py`, and `plugins/plugin_system.py` shed their dynamic IR access: contexts expose typed `function_registry`/`ir_functions` accessors, const var handling uses real `aval` attributes, and function exports rely on native `onnx_ir.Function` identifiers.
    - ✅ Further trimmed `conversion_api` dynamic paths: attr construction now leans on `onnx_ir.Attr`, function export uses `identifier()`, and value-info shape reconciliation prefers the typed IR context helpers (with proto fallbacks).
    - ✅ `_finalize_model_value_info_shapes` and attr override paths now operate on typed `ir.Value`/`ir.Graph` containers (no proto mirrors), and Concat axis defaults rely on native attribute assignment.
  - `scripts/audit_ir_dynamic_access.py` reports dynamic `getattr` usage; wire it into CI once the baseline shrinks to something manageable.
  - Drift guard: extend `tests/extra_tests/framework/test_onnx_ir_surface.py` (or equivalent) so it asserts the upstream package still exposes the attributes we depend on.
  - First cleanups landed in `converter/ir_pretty.py`, `converter/ir_postprocess.py`, `converter/ir_optimizations.py` (helpers + cast/transpose/dropout passes), and `plugins/_ir_shapes.py`; dynamic graph/value access now goes through typed properties/helpers.
  - TODO: sweep any remaining legacy helpers (e.g., attr fallbacks) and widen mypy coverage to the whole converter once the baseline is clean.
  - `_read_scalar_bool_from_value_or_constant` now relies on typed IR payloads, keeping dropout Not-elimination tests green.
  - ✅ Current state is stable; ready to resume in a follow-up session / new chat when continuing the cleanup.

## Raw Inputs
1. Setting a private field is not supported; Attr should be treated as immutable and replaced if modified.
2. Check for `None` before calling `attr.get_graphs()` to keep return types well defined.
3. Attr always defines its documented fields; using `getattr` is redundant.
4. Iterating attributes manually is redundant; just `return node.attributes.values()`.
5. Suggestion: `def _attribute_iter(node: ir.Node) -> Iterable[ir.Attr]:`.
6. Custom node collection is redundant; iterate directly on `Graph` or call `Graph.all_nodes()`.
7. Shape handling is overly complex; use `onnx_ir.Shape.is_unknown_dim`.
8. Remove pervasive `getattr` usage in favor of the correct IR APIs.
9. Node inputs are always defined (outside legacy proto), so extra guards are unnecessary.
10. Use `ir.convenience.replace_all_uses_with` for input replacement.
11. Suggestion: `def _clone_shape_obj(shape): return ir.Shape(shape)`.
12. Question raised about purpose of a helper; intent is unclear.
13. Prefer the existing helper in `ir.convenience` instead of custom logic.
14. A complex helper should be simplified; reviewer requested more context on intent.
15. Built-ins exist: use `value.producer()` / `value.consumers()`; nodes have `predecessors()` / `successors()`.
16. No need for a custom consumer map; `value.consumers()` handles it.
17. Use `value.uses()` instead of recreating that logic.
18. Avoid redundant attribute retrieval logic; rely on documented APIs for values.
19. Reference: `onnx_ir._graph_containers.Attributes` already provides needed accessors.
20. Use `value.dtype` instead of manual dtype extraction.
21. Prefer onnxscript optimizer passes for the optimization work.
22. Drop bespoke manipulation logic; lean on `ir.convenience` helpers.
23. Explicit recommendation: pair `onnxscript.optimizer.fold_constants.FoldConstantsPass` with `LiftConstantsToInitializersPass`; review IR passes in docs.
24. Overall request: remove try/except and `getattr` strewn through builder/context code; rely on typed APIs.
25. You may refer to https://github.com/onnx/ir-py/blob/59086bc749b10ff690578ebfe4b8527f38b89517/src/onnx_ir/passes/common/identity_elimination.py#L108-L121 to see how outputs are properly handled—use the native helpers and drop the proto mirrors entirely.
26. Identity elimination shows the pattern: check `Value.is_graph_input/output`, call `ir.convenience.replace_all_uses_with`, rename the survivor, update `graph.outputs` with Value objects, and remove the node via `graph.remove(..., safe=True)`.

## Rules Learned
- R1: Treat IR objects (Attr, Value, Graph) as typed APIs—avoid `getattr`, private field mutation, or manual guards (inputs 1, 2, 3, 8, 9, 18, 19, 24).
- R2: When traversing graphs or values, use built-in iterators and convenience helpers instead of reimplementing collections (inputs 4, 6, 10, 15, 16, 17, 22).
- R3: For shape and dtype logic, prefer provided helpers like `Shape.is_unknown_dim`, `ir.Shape(...)`, and `value.dtype` (inputs 7, 11, 20).
- R4: Before adding bespoke passes or rewrites, check onnxscript optimizer passes and documented IR passes (inputs 21, 23).
- R5: Keep helpers simple and well explained; collapse overly complex utilities or document their intent (inputs 12, 13, 14).
- R6: Maintain clear signatures and typing aligned with IR classes (input 5) to catch API drift early.
- R7: Prefer `onnx_ir`'s built-in helpers (Attr/AttributeType, Function.identifier, live Graph containers) and remove proto shims now that the typed APIs are always available (input 25).
- R8: When rewiring graph outputs, treat them as `Value` references—use `is_graph_output`, rename the survivor Value, replace entries in `graph.outputs`, and drop the node with `graph.remove(..., safe=True)` (input 26).

## Apply Rules
- R1: Audit converter and plugin code to replace `getattr`/private field mutations with direct API calls (`attr.get_graphs()`, `value.dtype`, etc.) and rebuild objects when necessary.
- R2: Standardize on `for node in graph`, `graph.all_nodes()`, `value.producer()`, `value.consumers()`, and `ir.convenience.replace_all_uses_with` for traversal and rewiring.
- R3: Replace homemade shape/dtype helpers with `Shape.is_unknown_dim`, `ir.Shape(shape)` cloning, and related utilities; delete redundant checks.
- R4: Integrate onnxscript optimizer passes (e.g., FoldConstants + LiftConstants) and IR common passes before writing new ones.
- R5: Review complex helpers, document their goal, or refactor into smaller, clearer pieces; ensure reviewers understand intent.
- R6: Update helper signatures to use concrete IR types (`Iterable[ir.Attr]`, etc.) and rely on static typing instead of dynamic guards.
- R7: When constructing attrs, exporting functions, or iterating graphs, call the canonical `onnx_ir` methods and excise any leftover proto mirrors.
- R8: Mirror the identity-elimination pattern for rewrites: use `replace_all_uses_with`, rename surviving Values, and update `graph.outputs` in-place before removing the original node with `safe=True`.

## Answers to Inputs
1. Follow R1: stop setting private Attr fields; rebuild Attr instances when changes are needed.
2. Follow R1: introduce explicit `None` checks before calling `attr.get_graphs()`.
3. Follow R1/R6: replace `getattr` with direct field access since Attr guarantees those members.
4. Follow R2: return `node.attributes.values()` directly.
5. Follow R6: update `_attribute_iter` signature to return `Iterable[ir.Attr]`.
6. Follow R2: drop custom node gathering; iterate graph or call `graph.all_nodes()`.
7. Follow R3: use `Shape.is_unknown_dim` for the unknown-dimension check.
8. Follow R1: remove generic `getattr` usage in the module and rely on typed access.
9. Follow R1/R2: assume `node.inputs` exists; remove redundant guards.
10. Follow R2: call `ir.convenience.replace_all_uses_with` for rewiring instead of manual loops.
11. Follow R3: implement `_clone_shape_obj` as `return ir.Shape(shape)`.
12. Follow R5: clarify or simplify the helper; document what it does.
13. Follow R5/R2: swap in the corresponding `ir.convenience` helper.
14. Follow R5: simplify the helper and explain its intent to the reviewer.
15. Follow R2: use `value.producer()` and `value.consumers()` (and node predecessors/successors) instead of custom bookkeeping.
16. Follow R2: drop consumer maps; rely on `value.consumers()`.
17. Follow R2: use `value.uses()` for usage counts or iteration.
18. Follow R1: replace redundant attribute retrieval with API calls from the documentation.
19. Follow R1/R2: depend on `_graph_containers.Attributes` accessors rather than manual wiring.
20. Follow R3: use `value.dtype` instead of manual dtype inference.
21. Follow R4: apply onnxscript optimizer passes rather than hand-rolled equivalents.
22. Follow R2/R4: defer to `ir.convenience` methods for graph manipulation.
23. Follow R4: adopt the recommended optimizer pass pairing and study IR pass docs before custom work.
24. Follow R1/R6: remove try/except and `getattr` scaffolding; rely on typed IR APIs throughout context and builder code.
25. Follow R7: default to `onnx_ir` helpers (Attr, AttributeType, Function.identifier, Graph.nodes) and only fall back to proto mirrors when necessary.
26. Follow R8: when eliminating nodes that feed graph outputs, swap consumers via `replace_all_uses_with`, rename the surviving Value, rewrite the `graph.outputs` list, and remove the node safely.
