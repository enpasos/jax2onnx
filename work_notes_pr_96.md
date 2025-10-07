# PR 96 Review Notes

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

## Rules Learned
- R1: Treat IR objects (Attr, Value, Graph) as typed APIs—avoid `getattr`, private field mutation, or manual guards (inputs 1, 2, 3, 8, 9, 18, 19, 24).
- R2: When traversing graphs or values, use built-in iterators and convenience helpers instead of reimplementing collections (inputs 4, 6, 10, 15, 16, 17, 22).
- R3: For shape and dtype logic, prefer provided helpers like `Shape.is_unknown_dim`, `ir.Shape(...)`, and `value.dtype` (inputs 7, 11, 20).
- R4: Before adding bespoke passes or rewrites, check onnxscript optimizer passes and documented IR passes (inputs 21, 23).
- R5: Keep helpers simple and well explained; collapse overly complex utilities or document their intent (inputs 12, 13, 14).
- R6: Maintain clear signatures and typing aligned with IR classes (input 5) to catch API drift early.

## Apply Rules
- R1: Audit converter and plugin code to replace `getattr`/private field mutations with direct API calls (`attr.get_graphs()`, `value.dtype`, etc.) and rebuild objects when necessary.
- R2: Standardize on `for node in graph`, `graph.all_nodes()`, `value.producer()`, `value.consumers()`, and `ir.convenience.replace_all_uses_with` for traversal and rewiring.
- R3: Replace homemade shape/dtype helpers with `Shape.is_unknown_dim`, `ir.Shape(shape)` cloning, and related utilities; delete redundant checks.
- R4: Integrate onnxscript optimizer passes (e.g., FoldConstants + LiftConstants) and IR common passes before writing new ones.
- R5: Review complex helpers, document their goal, or refactor into smaller, clearer pieces; ensure reviewers understand intent.
- R6: Update helper signatures to use concrete IR types (`Iterable[ir.Attr]`, etc.) and rely on static typing instead of dynamic guards.

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
