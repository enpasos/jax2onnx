# IR Reflection & Typed API Guidelines

These notes capture the durable lessons from the reflection cleanup completed in PR 96. Use them when extending the converter, optimizer, or plugin helpers so we stay aligned with the typed `onnx_ir` APIs and keep future audits small.

## Cleanup Workflow

17. Introduce or reuse typed utility helpers (e.g. leveraging `val.dtype` or `val.shape`) and migrate one module at a time instead of broad rewrites.  
2. Run `scripts/audit_ir_dynamic_access.py` after each sweep. Classify remaining hits as either *expected* (document why) or *needs follow-up*.  
3. Re-run `mypy` together with a focused `pytest` slice before marking the sweep complete.

Document ONNX proto shims that still require reflection (`plugins/_post_check_onnx_graph.py`, `plugins/_patching.py`) so the audit trail stays actionable.

## Core Rules

- **Typed APIs first** — Treat `ir.Attr`, `ir.Value`, and `ir.Graph` as immutable, typed objects. Avoid `getattr`, private field mutation, or redundant `None` guards; rely on the documented accessors instead.
- **Use built-in graph iterators** — Iterate via `graph.nodes`, `graph.all_nodes()`, `value.consumers()`, `value.producer()`, and `ir.convenience.replace_all_uses_with` instead of constructing custom maps.
- When working with `onnx_ir` graphs, prefer `graph.all_nodes()` so nested functions and control-flow subgraphs are traversed with the typed iterator. Only fall back to the ONNX proto mirrors (`function.node`) when you truly need the raw proto objects.
- **Reuse shape & dtype helpers** — Use `onnx_ir.Shape.is_unknown_dim`, `ir.Shape(...)`, and `value.dtype` rather than cloning dtype/shape logic manually.
- **Prefer existing passes** — Before adding bespoke optimizations, check the available ONNX Script passes (e.g., `fold_constants.FoldConstantsPass`, `LiftConstantsToInitializersPass`) and the IR optimizer docs.
- **Keep helpers focused & typed** — Provide clear function signatures (e.g., `_attribute_iter(node: ir.Node) -> Iterable[ir.Attr]`) and avoid over-engineered utilities. Document non-obvious helpers.
- **Lean on `onnx_ir` containers** — Use live `Attributes`, `Functions`, and other containers instead of proto mirrors; drop shims now that typed APIs are universal.
- **Graph rewrites** — When eliminating nodes (e.g., identity removal), check `Value.is_graph_input/output`, rename the surviving value, update `graph.outputs` with `Value` objects, call `ir.convenience.replace_all_uses_with`, and remove the node via `graph.remove(..., safe=True)`.
- **Annotate aggressively** — Keep explicit type annotations in converter and plugin code so mypy enforces alignment with IR interfaces.
