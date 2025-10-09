# PR 96 Review Notes

- Baseline cleanup is done; the remaining reflection work should target real wins, not a blanket “ban `getattr`”.
- Primary focus areas:
  * `converter/ir_context.py`: replace the noisy `getattr(var, "aval", ...)` branches with small typed helpers; keep literal fallbacks where they protect edge cases.
  * `converter/ir_optimizations.py`: audit the attr/shape helpers next and tighten signatures so audit output for that module shrinks.
  * `plugins/_post_check_onnx_graph.py` and `_patching.py`: these still need reflection for ONNX proto shims—document that expectation so the audit report is actionable.
- Workflow for each sweep:
  1. Add typed utility functions (e.g., `_maybe_aval`, `_maybe_dtype`) and migrate one module at a time.
  2. Run `scripts/audit_ir_dynamic_access.py`; classify remaining hits as “expected” vs. “needs follow-up”.
  3. Re-run mypy + a focused pytest slice before marking the task complete.
- Reminder: keep this note lean—record only actionable follow-ups.  




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
- R9: Always provide explicit variable type annotations in backend processing so mypy enforces the converter/plugin interfaces.


