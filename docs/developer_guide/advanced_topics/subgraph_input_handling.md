# Control-Flow Subgraph Inputs (ONNX)

Our IR-only pipeline still emits ONNX control-flow nodes, so plugin authors and optimizer work must respect the schema-defined subgraph contracts. Use this guide when touching `converter/ir_context.py`, the control-flow plugins, or the structural tests that verify body graphs (for example `tests/extra_tests/scan/`).

## TL;DR
- `If` exposes exactly one explicit input (`cond`). Branch graphs declare zero formal inputs and capture any extra tensors by name from the parent scope.
- `Loop` forwards everything explicitly: `iteration_num`, `condition`, and N loop-carried values are provided to the body each iteration. Thread outer tensors through the carried tuple when they are not constants.
- `Scan` mirrors `Loop`, adding per-step slices for each scan input sequence. Body inputs are ordered as state variables followed by sequence slices; nothing is implicitly captured.

## `If`: implicit capture only
The ONNX `If` node accepts a single boolean input. Both `then_branch` and `else_branch` Graph attributes must list zero inputs, so any tensor needed inside the branch is referenced directly by name and resolved from the enclosing graph. Exporters that try to add branch parameters fail schema validation (`input size of if-op should be 1`). Our lowering honours that contract—branch builders leave the input list empty and rely on `IRContext` to resolve captured values.

## `Loop`: explicit interface
`Loop` nodes receive an optional trip-count `M`, an optional initial condition, and N loop-carried initial values. Consequently, the body graph declares `iteration_num`, `condition`, and those N carried values as explicit inputs. Each iteration returns a continuation flag, the updated carried values, and optional scan outputs that map back to the parent node. When the body needs an outer tensor that is not constant, add it to the carried tuple (passing it through unchanged if necessary). This keeps dependencies explicit and aligns with ONNX Runtime’s validation. Our `fori_loop` and `scan` plugins rely on this ordering when constructing body graphs.

## `Scan`: explicit interface with sequence slices
`Scan` behaves like `Loop` with additional scan inputs. The body graph lists N state variables followed by M per-iteration slices—one from each scan input tensor. It yields updated state plus K scan outputs that the runtime stacks. Every non-constant value must arrive through these inputs; there is no implicit capture. Thread outer tensors as state variables if you need them on each step. Tests under `tests/extra_tests/scan/` assert these invariants so regressions surface quickly.

## Practical tips
- `IRContext.get_value_for_var` is responsible for materialising captured tensors. Keep its literal handling consistent with these rules.
- When authoring tests, prefer `expect_graph` assertions on body graphs to ensure input/output arity matches the spec.
- If ONNX adjusts the schemas, update the converter code and this guide together so plugin authors retain a single source of truth.

## Constants inside subgraphs (no initializers)
- ONNX Functions and control‑flow subgraphs must not contain graph initializers. All constants inside `Loop`/`Scan` bodies or Function graphs are emitted as `Constant` nodes.
- Our converter enforces this by running subgraph construction in “function mode”, which makes builder initializer helpers produce `Constant` nodes instead of registering initializers on the body graph.
- Plugin authors should always use `ctx.builder.add_initializer_from_*` or `ctx.bind_const_for_var(...)` for constants so the correct form is emitted automatically in subgraphs. Avoid writing to any `_initializers` lists directly.
- Post‑processing loosens shapes inside subgraphs: value shapes in Loop/Scan bodies are set to rank‑only (all dims unknown) to reduce schema friction and improve portability. Structural tests assert this behaviour.
